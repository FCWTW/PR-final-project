import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine import Config
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = '/home/wayne/Desktop/PR_final/test_img/original.png'
OUTPUT_PATH = '/home/wayne/Desktop/PR_final/result/segformer_img.png'
CONFIG_PATH = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'

def save_segmentation_result(result, class_names):
    # Extract masks and generate color images
    pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
    max_index = np.max(pred_mask)
    # print(f"Max class index in pred_mask: {max_index}")
    
    # Generate high-contrast colors using matplotlib colormap
    cmap = plt.get_cmap('tab20')
    palette = (cmap(np.linspace(0, 1, len(class_names)))[:, :3] * 255).astype(np.uint8)

    if max_index >= len(palette):
        raise ValueError(f"Palette size is insufficient for class index {max_index}")
    
    # Create colorful mask
    color_seg = palette[pred_mask]
    input_image = cv2.imread(IMAGE_PATH)
    seg_map = cv2.addWeighted(input_image, 0.5, color_seg, 0.5, 0)
    cv2.imwrite(OUTPUT_PATH, seg_map)

if __name__ == "__main__":
    # Prepare for segformer
    cfg = Config.fromfile(CONFIG_PATH)
    model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')
    class_names = model.dataset_meta['classes']

    result = inference_model(model, IMAGE_PATH)
    save_segmentation_result(result, class_names)