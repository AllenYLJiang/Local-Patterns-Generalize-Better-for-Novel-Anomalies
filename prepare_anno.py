import os 
import shutil
import json
import joblib

# this is the concatenation of 17-channel heatmaps and 3-channel human images
heatmap_dir = 'backbone/heatmaps_resnet'

anno_list = []
image_id = 1
for img_tensor_name in os.listdir(heatmap_dir):
    print(img_tensor_name)
    if len(img_tensor_name.split('_' + img_tensor_name.split('.pkl')[0].split('_')[-1] + '.pkl')[0]) == 7:
        continue

    curr_img_contents = {
        'image': os.path.join(heatmap_dir, img_tensor_name),
        'caption': '',
        'image_id': 'hrnet_' + str(image_id),
        'dataset': 'hrnet'
    }
    anno_list.append(curr_img_contents)
    image_id += 1

out_file = open("resnet_caption.json", "w")
json.dump(anno_list, out_file)
out_file.close()