import cv2
import numpy as np
from networks.ccnet import Seg_Model
import torch
import torch.nn as nn

NUM_CLASSES = 19
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
INPUT_SIZE = (769, 769)

MODEL_PATH = '/media/wayne/614E3B357F566CB2/model/CCNetR1.pth'
IMAGE_PATH = '/home/wayne/Desktop/PR_final/test_img/original.png'
OUTPUT_PATH = '/home/wayne/Desktop/PR_final/result/ccnet.png'

def get_palette(num_cls):
    palette = np.zeros((num_cls, 3), dtype=np.uint8)
    for j in range(num_cls):
        lab = j
        palette[j, 0] = ((lab >> 0) & 1) * 255
        palette[j, 1] = ((lab >> 1) & 1) * 255
        palette[j, 2] = ((lab >> 2) & 1) * 255
    return palette


def predict_whole(net, image, tile_size):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    with torch.no_grad():
        prediction = net(image)
        if isinstance(prediction, list):
            prediction = prediction[0]
        prediction = interp(prediction).cpu().numpy()[0]  # (19, H, W)
    return prediction
# def predict_whole(net, image, tile_size, flip_evaluation, recurrence):
#     interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
#     prediction = net(image.cuda(), recurrence)
#     if isinstance(prediction, list):
#         prediction = prediction[0]
#     prediction = interp(prediction).cpu().data[0].numpy().transpose(1,2,0)
#     return prediction

if __name__ == '__main__':
    print("Load model")
    # Load model
    model = Seg_Model(num_classes=NUM_CLASSES, pretrained_model=MODEL_PATH)
    model.eval().cuda()

    print("Process image")
    # Load image
    image = cv2.imread(IMAGE_PATH).astype(np.float32)
    h, w, _ = image.shape

    # Image preprocessing
    image -= IMG_MEAN  # 減去均值標準化
    image = cv2.resize(image, INPUT_SIZE)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image).unsqueeze(0).cuda()

    print("testing...")
    # Predict
    output = predict_whole(model, image, (h, w))
    print("finish test")
    # Get segmentation map
    seg_map = np.argmax(output, axis=0).astype(np.uint8)

    # Create colorful mask
    palette = get_palette(NUM_CLASSES)
    color_seg_map = palette[seg_map]
    color_seg_map = cv2.resize(color_seg_map, (w, h))
    overlay = cv2.addWeighted(cv2.imread(IMAGE_PATH), 0.5, color_seg_map, 0.5, 0)

    # Save result
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Finish")