import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine import Config
import cv2

def save_segmentation_result(result, img_path, save_path):

    input_image = cv2.imread(img_path)

    # Extract masks and generate color images
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()  # Assuming a Segmentation Map is returned
    max_index = np.max(pred_mask)
    print(f"Max class index in pred_mask: {max_index}")
    
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0], [0, 64, 64], [64, 0, 64], [64, 64, 64],
        [192, 0, 0], [0, 192, 0], [0, 0, 192], [192, 192, 0], [0, 192, 192], [192, 0, 192], [192, 192, 192]
    ])

    if max_index >= len(palette):
        raise ValueError(f"Palette size is insufficient for class index {max_index}")
    
    color_seg = palette[pred_mask].astype(input_image.dtype)
    
    # print(f"Input image dtype: {input_image.dtype}")
    # print(f"Color segmentation dtype: {color_seg.dtype}")

    overlay = cv2.addWeighted(input_image, 0.5, color_seg, 0.5, 0)
    cv2.imwrite(save_path, overlay)

def main(config_file, img_path):
    # load config file
    cfg = Config.fromfile(config_file)

    # initialize model
    model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')

    # inference
    result = inference_model(model, img_path)
    save_path = ''
    save_segmentation_result(result, img_path, save_path)

if __name__ == '__main__':
    config_file = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'
    img_path = ''

    # cfg = Config.fromfile(config_file)
    # print(cfg.pretty_text)

    main(config_file, img_path)