import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import cv2
import numpy as np
import joblib
import json
import random
import torch.nn.functional as F
import os

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True,
                                                     device=device)  # , torch_dtype=torch.float16)

#########################################################################################################################################################
####################################### Test ############################################################################################################
#########################################################################################################################################################
dst_dir = 'Image_Module_output_features_test'

test_data_index_file = json.load(open('resnet_caption_test.json'))
# sampled_file = test_data_index_file[int(len(test_data_index_file) * random.random())]
batch_size = 100
image_list, save_dir_list = [], []
for test_set_idx in range(len(test_data_index_file)):
    sampled_file = test_data_index_file[test_set_idx]
    random_sampled_img = joblib.load(sampled_file['image'])
    gt_caption = sampled_file['caption']
    # cv2.imwrite('/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/dump.png', random_sampled_img[:, :, -3:])
    # load sample image
    # raw_image = Image.open("/root/Downloads/blip2_inference_demo/164_cropped.jpg").convert("RGB")
    # image = np.swapaxes(np.swapaxes(random_sampled_img[:, :, -3:], 1, 2), 0, 1)
    # image = np.expand_dims(image, axis=0)
    # image = torch.from_numpy(image).to(device)
    # image = F.interpolate(image, size=(128, 128), mode='nearest')
    image = random_sampled_img[512:768, :, :]
    image = torch.from_numpy(image).to(device)
    image_height, image_width, image_channel = image.shape[1], image.shape[2], image.shape[0]
    image = image.view(-1, image_channel, image_height, image_width)
    image = F.interpolate(image, size=(32, 32), mode='nearest')#[0]

    # prepare the image
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) #, torch.float16)

    if test_set_idx > 0 and (test_set_idx + 1) % batch_size == 0:
        image_list.append(image)
        save_dir_list.append(os.path.join(dst_dir, sampled_file['image'].split('/')[-1]))
        cross_attention_output_features = model.forward_image(torch.cat(image_list, axis=0))[0].cpu().detach().numpy()
        for idx_inside_batch in range(batch_size):
            joblib.dump(cross_attention_output_features[idx_inside_batch, :, :], save_dir_list[idx_inside_batch])
        image_list, save_dir_list = [], []
    else:
        image_list.append(image)
        save_dir_list.append(os.path.join(dst_dir, sampled_file['image'].split('/')[-1]))

    # answer = model.generate({"image": image, "prompt": " Question: what is the posture of the man in the video? "})

#########################################################################################################################################################
###################################################### Train ############################################################################################
#########################################################################################################################################################
dst_dir = 'Image_Module_output_features_train'

test_data_index_file = json.load(open('resnet_caption.json'))
# sampled_file = test_data_index_file[int(len(test_data_index_file) * random.random())]
batch_size = 100
image_list, save_dir_list = [], []
for test_set_idx in range(len(test_data_index_file)):
    sampled_file = test_data_index_file[test_set_idx]
    random_sampled_img = joblib.load(sampled_file['image'])
    gt_caption = sampled_file['caption']
    # cv2.imwrite('/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/dump.png', random_sampled_img[:, :, -3:])
    # load sample image
    # raw_image = Image.open("/root/Downloads/blip2_inference_demo/164_cropped.jpg").convert("RGB")
    # image = np.swapaxes(np.swapaxes(random_sampled_img[:, :, -3:], 1, 2), 0, 1)
    # image = np.expand_dims(image, axis=0)
    # image = torch.from_numpy(image).to(device)
    # image = F.interpolate(image, size=(128, 128), mode='nearest')
    image = random_sampled_img[512:768, :, :]
    image = torch.from_numpy(image).to(device)
    image_height, image_width, image_channel = image.shape[1], image.shape[2], image.shape[0]
    image = image.view(-1, image_channel, image_height, image_width)
    image = F.interpolate(image, size=(32, 32), mode='nearest')#[0]

    # prepare the image
    # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device) #, torch.float16)

    if test_set_idx > 0 and (test_set_idx + 1) % batch_size == 0:
        image_list.append(image)
        save_dir_list.append(os.path.join(dst_dir, sampled_file['image'].split('/')[-1]))
        cross_attention_output_features = model.forward_image(torch.cat(image_list, axis=0))[0].cpu().detach().numpy()
        for idx_inside_batch in range(batch_size):
            joblib.dump(cross_attention_output_features[idx_inside_batch, :, :], save_dir_list[idx_inside_batch])
        image_list, save_dir_list = [], []
    else:
        image_list.append(image)
        save_dir_list.append(os.path.join(dst_dir, sampled_file['image'].split('/')[-1]))









# model.generate({"image": image, "prompt": "Question: What action? Answer: "})

# cv2.imwrite('/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/dump.png', np.swapaxes(np.swapaxes(image[0], 0, 1), 1, 2).cpu().numpy())
# AAAA=np.swapaxes(np.swapaxes(image.cpu().numpy()[0], 0, 1), 1, 2)
# AAAA=AAAA - np.min(AAAA)
# AAAA=AAAA / np.max(AAAA) * 255
# cv2.imwrite('/root/data/Anomaly_detection_921/STG-NF/data/ShanghaiTech/images/image_hrnet_blip2caption/dump.png', AAAA)
