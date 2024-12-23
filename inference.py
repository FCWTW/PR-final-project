import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine import Config
import cv2
import sys
import matplotlib.pyplot as plt

video_url = '/home/wayne/Desktop/PR_final/test_video/001.mp4'
config_file = '/home/wayne/Desktop/PR_final/segformer/config_b5.py'

def segmentation_result(result, input_image, class_names):
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
    color_seg = palette[pred_mask].astype(input_image.dtype)
    seg_map = cv2.addWeighted(input_image, 0.5, color_seg, 0.5, 0)

    used_classes = np.unique(pred_mask)
    seg_map_with_labels = draw_labels(seg_map, class_names, used_classes, palette)
    return seg_map_with_labels

# Draw class names and corresponding colors below the image
def draw_labels(image, class_names, used_classes, palette):
    canvas_height = image.shape[0] + 300
    canvas = np.ones((canvas_height, image.shape[1], 3), dtype=np.uint8) * 255

    # Copy the image onto the canvas
    canvas[:image.shape[0], :, :] = image

    # Initialize offset
    font_scale = 1.5
    font_color = (0, 0, 0)
    label_box_size = 60  # (font_scale * 20) and label box is a square
    y_offset = image.shape[0] + 20
    x_offset_start = 60
    x_offset = x_offset_start

    # Draw each label
    for i, class_idx in enumerate(used_classes):
        label = class_names[class_idx]
        color = palette[class_idx].tolist()

        if y_offset + label_box_size <= canvas_height:
            cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + label_box_size, y_offset + label_box_size), color, -1)
            cv2.putText(canvas, label, (x_offset + label_box_size + 10, y_offset + label_box_size - 12), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 4)
            x_offset += 350  # Adjust spacing
        else:
            print("Error: Out of bounding.")
            sys.exit(1)

        # Put only six categories per row
        if (i + 1) % 5 == 0:
            x_offset = x_offset_start
            y_offset += label_box_size + 10

    return canvas

if __name__ == "__main__":
    # Prepare for segformer
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint=cfg.model.backbone.init_cfg.checkpoint, device='cuda:0')
    class_names = model.dataset_meta['classes']

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_url}")
        sys.exit(1)
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from video.")
        sys.exit(1)

    # Calculate output size
    height, width = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('./result/001.mp4', fourcc, fps, (width, height + 300))

    # Process video
    while(ret):
        result = inference_model(model, frame)
        seg_map = segmentation_result(result, frame, class_names)
        video_out.write(seg_map)
        ret, frame = cap.read()

    cap.release()
    video_out.release()