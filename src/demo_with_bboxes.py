# Some basic setup:
# Setup detectron2 logger
import detectron2
import json
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
import torch

def get_category_info(annotation_file_path):
    with open(annotation_file_path) as f:
        data = json.load(f)
    categories = data['categories']
    thing_classes = [category['name'] for category in categories]
    return thing_classes
 
# Load the annotation file and extract category information
annotation_file_path_train = os.path.join(os.environ['DATA_PATH'], "coco/annotations/instances_train2017.json")
thing_classes = get_category_info(annotation_file_path_train)
MetadataCatalog.get("my_dataset").thing_classes = thing_classes
coco_metadata = MetadataCatalog.get("my_dataset")

# import Mask2Former project
from mask2former import add_maskformer2_config

im = cv2.imread(os.path.join('../images/test/CFR_1626.jpg'))
#im = cv2.imread(os.path.join('../images/test/collage.jpg'))
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("../configs/coco/instance-segmentation/swin/maskformer2_swin_base_384_bs16_50ep.yaml")
cfg.MODEL.WEIGHTS = 'output/model_final.pth'
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
outputs = predictor(im_rgb)

# Print out the outputs for debugging

# Check if 'instances' key is present in the outputs
if "instances" in outputs:
    instances = outputs["instances"].to("cpu")
    instances = instances[instances.scores > 0.5]

    # Print details of the instances
    print("Predicted Classes:", instances.pred_classes)
    print("Predicted Scores:", instances.scores)
    print("Predicted Boxes:", instances.pred_boxes)

    # Visualize the predictions
    v = Visualizer(im_rgb[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    instance_result = v.draw_instance_predictions(instances).get_image()
    
    # Display the result
    cv2.imshow('image', instance_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    #Save image
    cv2.imwrite('../images/collage.jpg', instance_result)
    '''
else:
    print("No instances found in the output")
