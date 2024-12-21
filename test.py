import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine import Config
import cv2
import sys

video_url = '/home/wayne/Desktop/PR_final/test_video/001.mp4'
config_file = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'

def segmentation_result(result, input_image):
    # Extract masks and generate color images
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
    max_index = np.max(pred_mask)
    # print(f"Max class index in pred_mask: {max_index}")
    
    palette = np.array([
        [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [0, 64, 0], [0, 0, 64], [64, 64, 0], [0, 64, 64], [64, 0, 64], [64, 64, 64],
        [192, 0, 0], [0, 192, 0], [0, 0, 192], [192, 192, 0], [0, 192, 192], [192, 0, 192], [192, 192, 192]
    ])

    if max_index >= len(palette):
        raise ValueError(f"Palette size is insufficient for class index {max_index}")
    
    # Create colorful mask
    color_seg = palette[pred_mask].astype(input_image.dtype)
    seg_map = cv2.addWeighted(input_image, 0.5, color_seg, 0.5, 0)
    return seg_map

if __name__ == "__main__":
    # Prepare for segformer
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')

    flag = 0
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_url}")
        sys.exit(1)
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video.")
        sys.exit(1)

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('./result/001.mp4', fourcc, 30, (width,height))

    # Process video
    while(ret):
        result = inference_model(model, frame)
        seg_map = segmentation_result(result, frame)
        video_out.write(seg_map)
        ret, frame = cap.read()

    cap.release()
    video_out.release()