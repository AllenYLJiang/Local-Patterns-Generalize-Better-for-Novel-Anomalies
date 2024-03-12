# Local-Patterns-Generalize-Better-for-Novel-Anomalies

For better debugging, the code is divided into four parts: Detection, Backbone feature extraction, Identifying language-informative local pattern, and State-machine module

# Detection 
Input videos are placed according to the format: input_data\ 
We have utilized RPN (https://github.com/microsoft/RegionCLIP) for locating bounding boxes, the boxes are organized according to the format:  
detection_results\test  
detection_results\train 

# Backbone 
We have provided the backbone for feature extraction at backbone\ 
The process: backbone\resnet_featuremap_as_backbone.py 
The features are saved at: backbone\heatmaps_resnet 

# Identifying language-informative local pattern
prepare_anno.py organizes the results in backbone\heatmaps_resnet for Image-attention Module   
The backbone features corresponding to train set are stored at resnet_caption.json with only normal behaviors  
The backbone features corresponding to test set are stored at resnet_caption_test.json with abnormal behaviors    
Then run Image-attention Module with inference_Image-attention-Module.py   
Saved results: 
The outputs for test set: Image_Module_output_features_test folder   
The outputs for training sest: Image_Module_output_features_train folder   




