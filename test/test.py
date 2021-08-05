import cv2
import numpy as np
import pandas as pd
import os
from fasterRCNN import vgg_back, rpn, detector, fasterRCNN
from dataSequence import category_name
from config import config


if __name__ == '__main__':

    input_shape = (512,512)
    n_classes = len(category_name)
    anchors = config.anchors
    n_anchors = config.n_anchors

    rpn_model, detection_model, entire_model = fasterRCNN(input_shape, n_classes, n_anchors)

    # test rpn
    rpn_model.load_weights("weights/rpn.h5")

    
