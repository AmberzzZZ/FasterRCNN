import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance


def aug_slice(img, boxes, labels, target_shape):
    # resize
    img = cv2.resize(img, target_shape).astype(np.uint8)

    # color jit
    img = random_color_jittering(img)

    # random flip
    img, boxes, labels = random_flip(img, boxes, labels)

    return img, boxes, labels


def random_color_jittering(img):
    MAX_LEVEL = 10.
    factor = 9 / MAX_LEVEL * 1.8 + 0.1
    transforms = random.choices([color, contrast, brightness, sharpness], k=2)
    transforms = set(transforms)
    img = img.copy()
    if random.uniform(0,1)>0.5:
        for T in transforms:
            img = T(img, factor)
    return img


def random_flip(img, boxes, labels):
    img = img.copy()
    boxes = boxes.copy()
    labels = labels.copy()
    if random.uniform(0,1)>0.5:      # x-axis flip
        img = img[:,::-1,:]
        if boxes.shape[0]:
            boxes[:,0::2] = 1 - boxes[:,2::-2]    # x1=1-x2
    # else:                            # y-axis flip
    #     img = img[::-1,:,:]
    #     boxes[:,1::2] = 1 - boxes[:,3::-2]    # y1=1-y2
    return img, boxes, labels


def color(img, factor):
    img = ImageEnhance.Color(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def contrast(img, factor):
    img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def brightness(img, factor):
    img = ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor)
    return np.array(img)


def sharpness(img, factor):
    img = ImageEnhance.Sharpness(Image.fromarray(img)).enhance(factor)
    return np.array(img)
