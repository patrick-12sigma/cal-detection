import os
import sys
import random
import numpy as np
# import cv2
if __name__ == '__main__':
    
    import matplotlib
    import matplotlib.pyplot as plt

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
from mrcnn.config import Config

import skimage

if __name__ == '__main__':
    
    # Root directory of the project
    ROOT_DIR = '/home/sky8/cal/'
    # ROOT_DIR2 = '/data1/share/shiky/Mask_RCNN'
    img_train = os.path.join(ROOT_DIR, 'train', 'images') 
    img_val = os.path.join(ROOT_DIR, 'validation', 'images') 

    mask_train = os.path.join(ROOT_DIR, 'train', 'masks') 
    mask_val = os.path.join(ROOT_DIR, 'validation', 'masks') 


    train_imglist = os.listdir(img_train)
    val_imglist = os.listdir(img_val)

    train_count = len(train_imglist)
    val_count = len(val_imglist)
    # width = 1280
    # height = 800
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class CalConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "calcification-resnet50"

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1
    
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (750 - 20) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 20 // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400
    

 
 

   
class CalDataset(utils.Dataset):
    """Create customized dataset class to load data"""
    def load_cal(self,count, data_dir, mask_dir, namelist):
        """load """
        
        #Add classes
        self.add_class("breast", 1, "calcification" )
        
        #Add images
        
        for i in range(count):
            src = namelist[i]
            
            name = src.split('_')
            full_name = '_'.join(name[0:4])
            mask_middle= full_name.split('.')[0]
            mask_name = mask_middle + '_MASK.jpg'
            mask_path = os.path.join(mask_dir, mask_name)
            
            self.add_image("breast", image_id = i, path = os.path.join(data_dir, full_name), mask_path = mask_path , full_name =full_name)
#     def load_image(self, image_id):
#         """load image from local path"""
        
#         info = self.image_info[image_id]
       
#         image = skimage.io.imread(info.path)
        
#         return image
        
    def load_mask(self, image_id):
        """load the mask image"""
        
        info = self.image_info[image_id]
        
       
        mask = skimage.io.imread(info['mask_path']).astype(bool)
        mask = mask[:,:,np.newaxis]
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index("calcification")])
        
        return mask, class_ids.astype(np.int32)
        
    def load_name(self, image_id):
        """load the image name"""
        info = self.image_info[image_id]
        
        return info['full_name']

if __name__ == '__main__':
    
    config = CalConfig()
    # Training dataset
    dataset_train = CalDataset()
    dataset_train.load_cal(753, img_train, mask_train, train_imglist)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CalDataset()
    dataset_val.load_cal(200, img_val, mask_val, val_imglist)
    dataset_val.prepare()


    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.

    mode = "imagenet-resnet50"

    if mode == "coco":
        model_path = "/home/sky8/cal/code/mask_rcnn_coco.h5"

    if mode == "imagenet-resnet50":
        model_path = "/home/sky8/cal/code/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

    if mode == "imagenet-resnet101":
        model_path = "/home/sky8/cal/code/resnet101_weights_tf.h5"
    # 
    elif mode == "last":
        model_path = model.find_last()[1]


    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)

    if mode == "coco":
        model.load_weights(model_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)



    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=20, 
                layers='heads')


    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=40, 
                layers="all")

    print("ok")
