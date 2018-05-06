import os
import sys
import random
import numpy as np
from imgaug import augmenters as iaa
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
from Maskcon import CalConfig, CalDataset


import skimage

# Root directory of the project
ROOT_DIR = '/home/sky8/cal/data/ddsm-mass/'
LOG_DIR = '/home/sky8/project/sky8/cal/'
# ROOT_DIR2 = '/data1/share/shiky/Mask_RCNN'
img_train = os.path.join(ROOT_DIR, 'train', 'images')
img_val = os.path.join(ROOT_DIR, 'validation', 'images')

mask_train = os.path.join(ROOT_DIR, 'train', 'masks')
mask_val = os.path.join(ROOT_DIR, 'validation', 'masks')

train_imglist = os.listdir(img_train)
val_imglist = os.listdir(img_val)

train_count = len(train_imglist)
val_count = len(val_imglist)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(LOG_DIR, "logs")

class Config(CalConfig):
    NAME = "mass-coco-noaug-all"

config = Config()
# Training dataset
dataset_train = CalDataset()
dataset_train.load_cal(train_count, img_train, mask_train, train_imglist)
dataset_train.prepare()

# Validation dataset
dataset_val = CalDataset()
dataset_val.load_cal(val_count, img_val, mask_val, val_imglist)
dataset_val.prepare()


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

mode = "coco"

if mode == "specific":
    #model_path = "/home/sky8/cal/logs/calcification-coco20180419T0119/mask_rcnn_calcification-coco_0040.h5"
    model_path = "/home/sky8/cal/logs/calcification20180419T0048/mask_rcnn_calcification_0040.h5"
    
elif mode == "coco":
    model_path = "/home/sky8/cal/code/model/mask_rcnn_coco.h5"

elif mode == "imagenet-resnet50":
    model_path = "/home/sky8/cal/code/model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

elif mode == "imagenet-resnet101":
    model_path = "/home/sky8/cal/code/model/resnet101_weights_tf.h5"

elif mode == "none":
    pass

elif mode == "last":
    model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)

if mode == "coco":
    model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    print('loadok')
elif mode == "none":
    pass

else:
    model.load_weights(model_path, by_name=True)

print('load_weight')
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            augmentation=None,
            layers='heads')


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=60,
            augmentation=None,
            layers="all")

print("ok")
