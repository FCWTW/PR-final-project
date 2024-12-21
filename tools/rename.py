import os

# Rename image in dataset
base_folder = '/media/wayne/614E3B357F566CB2/cityscapes/gtFine/gtFine/train/'

for root, dirs, files in os.walk(base_folder):
    for filename in files:
        if filename.endswith('gtFine_labelIds.png'):
            # Get new name
            new_name = filename.replace('gtFine_labelIds', 'gtFine_labelTrainIds')
            # Get full file path
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_name)
            # Rename file
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_name}')

