# Local-Patterns-Generalize-Better-for-Novel-Anomalies

For better debugging, the code is divided into four parts: Detection, Backbone feature extraction, Identifying language-informative local pattern, and State-machine module

# Detection 
We have utilized RPN (https://github.com/microsoft/RegionCLIP) for locating bounding boxes, the boxes are organized according to the format:  
detection_results\test  
detection_results\train 

# Backbone 
We have provided the backbone for feature extraction at backbone\ 
The process: backbone\resnet_featuremap_as_backbone.py 
The features are saved at: backbone\heatmaps_resnet 


