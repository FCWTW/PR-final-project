from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count
import numpy as np
from networks.ccnet import Seg_Model
import torch
import cv2

NUM_CLASSES = 19
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
INPUT_SHAPE = (1, 3, 769, 769)
MODEL_PATH = '/media/wayne/614E3B357F566CB2/model/CCNetR1.pth'
IMAGE_PATH = '/home/wayne/Desktop/PR_final/test_img/original.png'

if __name__ == '__main__':
    # Load model
    print("Load model")
    model = Seg_Model(num_classes=NUM_CLASSES, pretrained_model=MODEL_PATH)
    model.eval().cuda()

    # Load image
    print("Process image")
    image = cv2.imread(IMAGE_PATH).astype(np.float32)
    h, w, _ = image.shape

    # Image preprocessing
    image -= IMG_MEAN
    image = cv2.resize(image, (769, 769))
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image).unsqueeze(0).cuda()

    # Calculate flops
    flops = FlopCountAnalysis(model, image)
    params = parameter_count(model)

    # Show result
    print(f"FLOPs: {flops.total()} | Params: {params['']}")

    # Show model information
    summary(model, input_size=INPUT_SHAPE)