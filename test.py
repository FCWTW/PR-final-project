'''
import os

data_root = '/media/wayne/614E3B357F566CB2/cityscapes/'
img_path = os.path.join(data_root, 'leftImg8bit/leftImg8bit/train')
seg_map_path = os.path.join(data_root, 'gtFine/gtFine/train')

print("Image path:", img_path)
print("Segmentation map path:", seg_map_path)

print("Image path exists:", os.path.exists(img_path))
print("Segmentation map path exists:", os.path.exists(seg_map_path))
'''

from mmseg.datasets import LoadAnnotations
import os
import numpy as np
from PIL import Image
import numpy as np

# test label
seg_map_path = '/media/wayne/614E3B357F566CB2/cityscapes/gtFine/gtFine/train/monchengladbach/monchengladbach_000000_026602_gtFine_labelIds.png'

img = Image.open(seg_map_path)
img_np = np.array(img)

# print(f"Image shape: {img_np.shape}")
# print(f"Unique pixel values: {np.unique(img_np)}")

if os.path.exists(seg_map_path):
    loader = LoadAnnotations(reduce_zero_label=False)
    results = dict(
        seg_map_path=seg_map_path,
        reduce_zero_label=False,
        seg_fields=[]
    )

    try:
        # 使用 PIL 打开文件
        img = Image.open(seg_map_path)
        label_map = np.array(img)
        print(f"Type of label_map: {type(label_map)}")
        # print(f"Label map shape: {label_map.shape}, Min: {label_map.min()}, Max: {label_map.max()}")
    except Exception as e:
        print(f"Error during manual loading: {e}")
        
    try:
        # Use loader
        results = loader(results)
        print(results)
        # Get label
        label_map = results.get('gt_seg_map', None)
        print(f"Type of label_map: {type(label_map)}")
        if isinstance(label_map, np.ndarray):
            print(f"Label map shape: {label_map.shape}, Min: {label_map.min()}, Max: {label_map.max()}")
        else:
            print("Failed to load label map.")
    except Exception as e:
        print(f"Error during loading: {e}")
else:
    print(f"Label file not found: {seg_map_path}")
