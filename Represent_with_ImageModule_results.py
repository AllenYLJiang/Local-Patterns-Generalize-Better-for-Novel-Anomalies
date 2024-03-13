import torch
import numpy as np
import os
import json
from PIL import Image
import cv2
import sys
import joblib
import time
import copy

###################### prepare images ######################################################################################
test_det_and_bbox_anno_dir = 'detection_results/test'
train_det_and_bbox_anno_dir = 'detection_results/train'
data_dir = [train_det_and_bbox_anno_dir, test_det_and_bbox_anno_dir]
frame_idx, overall_human_idx = 0, 0
dst_test_data_dir = 'State-Machine-Module/data/ShanghaiTech/pose/test'
dst_train_data_dir = 'State-Machine-Module/data/ShanghaiTech/pose/train'

src_heatmap_dir = 'Image_Module_output_features_train'
for data_split_dir in [train_det_and_bbox_anno_dir]:
    for video_json_name in os.listdir(data_split_dir): # [int(len(os.listdir(data_split_dir))/2 * 0): int(len(os.listdir(data_split_dir))/2 * 1)]:
        print("Current video: " + video_json_name)
        curr_video_json_content = json.load(open(os.path.join(data_split_dir, video_json_name)))
        video_name = video_json_name.split('_det-results.json')[0]
        for curr_frame_curr_human in curr_video_json_content:
            tmp_ele = copy.deepcopy(curr_frame_curr_human)
            tmp_ele['keypoints'] = [joblib.load(os.path.join(src_heatmap_dir, video_name + '_' + str(curr_video_json_content.index(curr_frame_curr_human)) + '.pkl'))]
            curr_video_json_content[curr_video_json_content.index(curr_frame_curr_human)] = tmp_ele
        joblib.dump(curr_video_json_content, os.path.join(dst_train_data_dir, video_json_name))

src_heatmap_dir = 'Image_Module_output_features_test'
for data_split_dir in [test_det_and_bbox_anno_dir]:
    for video_json_name in os.listdir(data_split_dir): # [int(len(os.listdir(data_split_dir))/2 * 0): int(len(os.listdir(data_split_dir))/2 * 1)]:
        print("Current video: " + video_json_name)
        curr_video_json_content = json.load(open(os.path.join(data_split_dir, video_json_name)))
        video_name = video_json_name.split('_det-results.json')[0]
        for curr_frame_curr_human in curr_video_json_content:
            tmp_ele = copy.deepcopy(curr_frame_curr_human)
            tmp_ele['keypoints'] = [joblib.load(os.path.join(src_heatmap_dir, video_name + '_' + str(curr_video_json_content.index(curr_frame_curr_human)) + '.pkl'))]
            curr_video_json_content[curr_video_json_content.index(curr_frame_curr_human)] = tmp_ele
        joblib.dump(curr_video_json_content, os.path.join(dst_test_data_dir, video_json_name))

