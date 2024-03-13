import torch
import numpy as np
import os
import json
from PIL import Image
from lavis.models import load_model_and_preprocess
import cv2
import sys
import joblib
from transformers import AutoImageProcessor, ResNetBackbone, ResNetForImageClassification
import torch.nn.functional as F
import time

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("D:\\Materials\\ECCV2024\\submit_code_updated\\backbone\\resnet")
model = ResNetBackbone.from_pretrained("D:\\Materials\\ECCV2024\\submit_code_updated\\backbone\\resnet", out_features=["stage2", "stage3", "stage4"]).to(device)

##################### load image preprocessor ##############################################################################
def draw_heatmaps_according_to_resnet(image):
    time_step1_start = time.time()
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        time_step1_end = time.time()
        inference_result = model(**inputs)
        time_step2_end = time.time()
        feature_maps_stage2, feature_maps_stage3, feature_maps_stage4 = inference_result.feature_maps[0], inference_result.feature_maps[1], inference_result.feature_maps[2]
        # logits = model(**inputs).logits
        feature_maps_stage2.requires_grad = False
        feature_maps_stage3.requires_grad = False
        feature_maps_stage4.requires_grad = False
        # feature_maps_stage2 = feature_maps_stage2[0]
        # feature_maps_stage3 = F.interpolate(feature_maps_stage3, size=(28, 28), mode='nearest')[0]
        batch_size_feature_maps_stage3, channels_feature_maps_stage3, height_feature_maps_stage3, width_feature_maps_stage3 = feature_maps_stage3.size()
        scale_vert_feature_maps_stage3, scale_hori_feature_maps_stage3 = 2, 2
        feature_maps_stage3 = feature_maps_stage3.view(batch_size_feature_maps_stage3,
                                                       int(channels_feature_maps_stage3/scale_vert_feature_maps_stage3/scale_hori_feature_maps_stage3),
                                                       height_feature_maps_stage3*scale_vert_feature_maps_stage3, width_feature_maps_stage3*scale_hori_feature_maps_stage3)

        # feature_maps_stage4 = F.interpolate(feature_maps_stage4, size=(28, 28), mode='nearest')[0]
        batch_size_feature_maps_stage4, channels_feature_maps_stage4, height_feature_maps_stage4, width_feature_maps_stage4 = feature_maps_stage4.size()
        scale_vert_feature_maps_stage4, scale_hori_feature_maps_stage4 = 4, 4
        feature_maps_stage4 = feature_maps_stage4.view(batch_size_feature_maps_stage4,
                                                       int(channels_feature_maps_stage4/scale_vert_feature_maps_stage4/scale_hori_feature_maps_stage4),
                                                       height_feature_maps_stage4*scale_vert_feature_maps_stage4, width_feature_maps_stage4*scale_hori_feature_maps_stage4)

        feature_maps_cat = torch.cat((feature_maps_stage2, feature_maps_stage3, feature_maps_stage4), dim=1)
        feature_maps_cat = feature_maps_cat.cpu().numpy()
        time_step3_end = time.time()
    return feature_maps_cat

###################### prepare images ######################################################################################
test_caption_and_bbox_anno_dir = 'D:\\Materials\\ECCV2024\\submit_code_updated\\detection_results\\test'
train_caption_and_bbox_anno_dir = 'D:\\Materials\\ECCV2024\\submit_code_updated\\detection_results\\train'
data_dir = [train_caption_and_bbox_anno_dir, test_caption_and_bbox_anno_dir]
img_dir = 'D:\\Materials\\ECCV2024\\submit_code_updated\\input_data'
dst_heatmap_dir = 'D:\\Materials\\ECCV2024\\submit_code_updated\\backbone\\heatmaps_resnet'

frame_idx, overall_human_idx = 0, 0
curr_human_list, curr_frame_curr_human_img_squared_list, curr_human_filename_list = [], [], []
for data_split_dir in data_dir:
    resume_index = 0 if (data_dir.index(data_split_dir) == 0) else 0
    for video_json_name in os.listdir(data_split_dir)[resume_index:]: # [int(len(os.listdir(data_split_dir))/2 * 0): int(len(os.listdir(data_split_dir))/2 * 1)]:
        print("Current video: " + video_json_name)
        curr_video_json_content = json.load(open(os.path.join(data_split_dir, video_json_name)))
        video_name = video_json_name.split('_det-results.json')[0]
        for curr_frame_curr_human in curr_video_json_content:
            frame_name = curr_frame_curr_human['image_id']
            # print("current frame: " + frame_name)
            # left top width height
            curr_human_bbox = curr_frame_curr_human['box']
            curr_human_bbox_left, curr_human_bbox_top, curr_human_bbox_width, curr_human_bbox_height = int(curr_human_bbox[0]), \
                                                                                                       int(curr_human_bbox[1]), \
                                                                                                       int(curr_human_bbox[2]), \
                                                                                                       int(curr_human_bbox[3])
            if 'test' in data_split_dir.split('/')[-1]: # video_name is the name of a folder
                curr_frame = cv2.imread(os.path.join(img_dir, video_name, frame_name))
            elif 'train' in data_split_dir.split('/')[-1]: # video_name is the name of an avi file
                cap = cv2.VideoCapture(os.path.join(img_dir, video_name+'.avi'))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_name.split('.')[0]))
                _, curr_frame = cap.read()
            curr_human = curr_frame[curr_human_bbox_top: curr_human_bbox_top + curr_human_bbox_height, \
                                    curr_human_bbox_left: curr_human_bbox_left + curr_human_bbox_width, :]

            # first hori, then vert coords
            curr_frame_curr_human_img_squared = np.zeros((max(curr_human.shape[:2]), max(curr_human.shape[:2]), curr_human.shape[2]), dtype='float32')
            if curr_human.shape[0] > curr_human.shape[1]:
                delta_half = int((curr_human.shape[0] - curr_human.shape[1]) / 2)
                curr_frame_curr_human_img_squared[:, delta_half:delta_half+curr_human.shape[1], :] = curr_human
            else:
                delta_half = int((curr_human.shape[1] - curr_human.shape[0]) / 2)
                curr_frame_curr_human_img_squared[delta_half:delta_half+curr_human.shape[0], :, :] = curr_human

            curr_human_list.append(curr_human)
            curr_frame_curr_human_img_squared_list.append(curr_frame_curr_human_img_squared)
            curr_human_filename_list.append(video_name + '_' + str(curr_video_json_content.index(curr_frame_curr_human)) + '.pkl')
            if overall_human_idx > 0 and (overall_human_idx+1) % 1024 == 0:
                curr_frame_curr_human_heatmaps = draw_heatmaps_according_to_resnet(curr_frame_curr_human_img_squared_list)
                # curr_frame_curr_human_heatmaps = np.swapaxes(np.swapaxes(curr_frame_curr_human_heatmaps, 0, 1), 1, 2)

                for inside_batch_idx in range(len(curr_human_list)):
                    joblib.dump(curr_frame_curr_human_heatmaps[inside_batch_idx, :, :, :], os.path.join(dst_heatmap_dir, curr_human_filename_list[inside_batch_idx]))

                curr_human_list, curr_frame_curr_human_img_squared_list, curr_human_filename_list = [], [], []

            overall_human_idx += 1


