import numpy as np
import cv2
import matplotlib.pyplot as plt

img_url = '/home/wayne/Desktop/PR_final/test_img/img.png'
gt_label_url = '/media/wayne/614E3B357F566CB2/cityscapes/gtFine/gtFine/test/berlin/berlin_000008_000019_gtFine_labelIds.png'
save_url = '/home/wayne/Desktop/PR_final/result/gt.png'

def visualize_ground_truth(gt_label_url, img_url, save_url):
    gt_label = cv2.imread(gt_label_url, cv2.IMREAD_UNCHANGED)
    if gt_label is None:
        raise FileNotFoundError(f"Cannot read {gt_label_url}")

    unique_labels = np.unique(gt_label)
    
    cmap = plt.get_cmap('tab20')
    palette = (cmap(np.linspace(0, 1, len(unique_labels)))[:, :3] * 255).astype(np.uint8)
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    color_seg = np.zeros((*gt_label.shape, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_seg[gt_label == label] = color

    input_image = cv2.imread(img_url)
    if input_image is None:
        raise FileNotFoundError(f"Cannot read {img_url}")

    if input_image.shape[:2] != gt_label.shape:
        raise ValueError("Ground Truth size does not match the input image size.")

    seg_map = cv2.addWeighted(input_image, 0.5, color_seg, 0.5, 0)
    
    cv2.imwrite(save_url, seg_map)
    print(f"Saved segmentation visualization at {save_url}")

if __name__ == "__main__":
    visualize_ground_truth(gt_label_url, img_url, save_url)