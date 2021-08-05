import numpy as np
import os
import cv2
import random
import math
import json
import pandas as pd
from keras.utils import Sequence
from config import config
from aug import aug_slice


category_name = ['cat', 'dog']
label_dict = {'cat': 0, 'dog': 1}
anchors = config.anchors
n_anchors = config.n_anchors

'''
gt: [b,M,c+1+4]
    feed as network inputs
    M: max boxes
    c+1: one-hot label, 1-channel is bg label instead of objectness
    4: normed box x1y1x2y2
'''
class dataSequence(Sequence):

    def __init__(self, img_dir, label_dir, num_classes, output_stride=16,
                 input_shape=None, batch_size=1, max_boxes=20):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_shape = input_shape[:2] if input_shape is not None else None
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.batch_size = batch_size if input_shape is not None else 1
        self.max_boxes = max_boxes

        self.anchors = anchors
        self.n_anchors = n_anchors
        # self.train = pd.read_pickle('train.pkl')   # [file_name, label]
        # print('full: ', len(self.train))
        self.train = [['tux_positive', '_']] #, ['tux_negative', '_'], ['tux_positive', '_'], ['tux_negative', '_']]   # ['tux_positive', '_'], 
        self.indices = np.arange(len(self.train))

    def __len__(self):
        return len(self.train) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_lst = [self.train[k] for k in batch_indices]
        x_batch, y_batch = self.data_generator(batch_lst)
        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def data_generator(self, batch_lst):
        print('current batch', batch_lst)
        n_classes = self.num_classes
        batch_size = self.batch_size
        anchors = self.anchors
        n_anchors = self.n_anchors
        anchors = np.reshape(anchors, (1,1,n_anchors,2))    # wh
        if self.input_shape is None and batch_size==1:
            file_name, _ = batch_lst[0]
            img, boxes, labels = get_img_boxes(self.img_dir, self.label_dir, file_name)
        else:
            target_shape = self.input_shape   # hw

        image_batch = []
        gt_batch = []
        for i in range(batch_size):
            if batch_size!=1 or self.input_shape is not None:
                file_name, _ = batch_lst[i]
                img, boxes, labels = get_img_boxes(self.img_dir, self.label_dir, file_name, target_shape)
            image_batch.append(np.expand_dims(img, axis=0))   # [1,h,w,3]
            gt = np.zeros((1,self.max_boxes, n_classes+1+4))
            n_boxes = boxes.shape[0]
            indices = np.arange(n_boxes)
            if n_boxes>self.max_boxes:
                np.random.shuffle(indices)
                indices = indices[:self.max_boxes]
            for idx, box_id in enumerate(indices):
                cls_id = int(labels[box_id])
                gt[0,idx,cls_id] = 1
                gt[0,idx,-4:] = boxes[box_id]
            gt_batch.append(gt)

        image_batch = np.concatenate(image_batch, axis=0)  # [b,h,w,3]
        # norm image
        if np.max(image_batch)>1:
            image_batch = image_batch / 255.
        gt_batch = np.concatenate(gt_batch, axis=0)   # [b,M,c+1+4]
        # add padding bg label
        gt_batch_bg = np.sum(gt_batch[...,:n_classes], axis=-1)
        gt_batch_bg = np.float32(gt_batch_bg==0)
        gt_batch[...,n_classes] = gt_batch_bg
        return [image_batch, gt_batch], np.zeros((batch_size))


def get_gt_offsets(anchors_all, boxes, labels, n_classes):
    hs, ws, n_anchors, _ = anchors_all.shape
    anchors_all = anchors_all.reshape((-1,4))   # [hwa,4], normed values
    print(boxes)
    iou, offset = cal_iou_offset(anchors_all, boxes)    # [hwa, N], [hwa, N, 4]
    gt = np.zeros((hs*ws*n_anchors, n_classes+1+4))
    gt[:,n_classes] = -1   # ignore

    # gt boxes' max match anchors: postive
    anchor_indices = np.argsort(iou.transpose(), axis=1)[:,-1]     # [N2,]
    for box_id, anchor_id in enumerate(anchor_indices):
        cls_id = int(labels[box_id])
        offsets = offset[anchor_id, box_id]
        gt[anchor_id][cls_id] = 1
        gt[anchor_id][n_classes] = 1
        gt[anchor_id][n_classes+1:] = offsets
    # maxIoU>0.7: positive
    max_iou = np.sort(iou)[:,-1]  # [N1]
    box_indices = np.argsort(iou)[:,-1]   # [N1], max matching box id, unique in [0,c)
    pos_coords = np.where(max_iou>0.7)[0]
    for anchor_id in pos_coords:
        box_id = int(box_indices[anchor_id])
        cls_id = int(labels[box_id])
        gt[anchor_id][cls_id] = 1
        gt[anchor_id][n_classes] = 1
        gt[anchor_id][n_classes+1:] = offset[anchor_id, box_id]
    # IoU<0.3: negtive
    neg_coords = np.where((max_iou<0.3) & (gt[...,n_classes]<1))[0]
    gt[neg_coords] = 0
    # reshape
    gt = np.reshape(gt, (hs,ws,n_anchors,n_classes+1+4))

    return gt


def get_img_boxes(img_dir, label_dir, file_name, target_shape=None):
    try:
        img = cv2.imread(os.path.join(img_dir, file_name+'.jpg'), 1)
        boxes = np.zeros((0))
        if os.path.exists(os.path.join(label_dir, file_name+'.json')):
            boxes = get_box(os.path.join(label_dir, file_name+'.json'))
            if boxes.shape[0]:
                labels = boxes[:,-1:]
                boxes = boxes[:,:-1]
        if not boxes.shape[0]:
            boxes = np.zeros((0))
            labels = []
        if target_shape is None:
            h, w, c = img.shape
            h, w = [int(h/w*600),600] if h>w else [600, int(w/h*600)]
            img, boxes, labels = aug_slice(img, boxes, labels, (w,h))
        else:
            img, boxes, labels = aug_slice(img, boxes, labels, target_shape)
    except:
        print('wrong img file', file_name)
        img = np.zeros((600,600,3)) if target_shape is None else np.zeros(target_shape+(3,))
        boxes = np.zeros((0))
        labels = []

    return img, boxes, labels     # [h,w,c], [N,4], [N,1]


def get_box(yolo_file):
    f = open(yolo_file, 'r')
    boxes = json.loads(f.read())
    box_arr = []
    for b in boxes:
        if b['label'] not in label_dict.keys():
            continue
        clsid = label_dict[b['label']]
        x1 = b['x1']
        x2 = b['x2']
        y1 = b['y1']
        y2 = b['y2']
        box_arr.append([x1,y1,x2,y2, clsid])
    f.close()
    return np.array(box_arr)   # [N,5], x1y1x2y2clsid, normed


def cal_iou_offset(boxes1, boxes2, epsilon=1e-5):
    # boxes: [N,4], x1y1x2y2, normed
    # return: [N1,N2], iou among boxes
    # return: [N1,N2,4], offsets between anchor_gt_pair
    boxes1 = boxes1.copy()
    boxes2 = boxes2.copy()
    boxes1 = np.expand_dims(boxes1, axis=1)   # [N1,1,4]
    boxes2 = np.expand_dims(boxes2, axis=0)   # [1,N2,4]

    inter_mines = np.maximum(boxes1[...,:2], boxes2[...,:2])    # [N1,N2,2]
    inter_maxes = np.minimum(boxes1[...,2:], boxes2[...,2:])
    inter_wh = np.maximum(inter_maxes - inter_mines, 0.)
    inter_area = inter_wh[...,0] * inter_wh[...,1]

    box_area1 = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
    # box_area1 = np.tile(box_area1, [1,np.shape(boxes2)[1]])
    box_area2 = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])
    # box_area2 = np.tile(box_area2, [np.shape(boxes1)[0],1])

    iou = inter_area / (box_area1 + box_area2 - inter_area + epsilon)
    print("maxiou", np.max(iou), iou.shape)

    wa = np.tile(boxes1[...,2]-boxes1[...,0], [1,boxes2.shape[1]])   # [N1,N2]
    ha = np.tile(boxes1[...,3]-boxes1[...,1], [1,boxes2.shape[1]])   # [N1,N2]
    tx = np.where(wa>0, (np.sum(boxes2[...,0::2],axis=-1)-np.sum(boxes1[...,0::2], axis=-1))/wa/2., np.zeros_like(wa))   # (x-xa)/wa, [N1,N2]
    ty = np.where(ha>0, (np.sum(boxes2[...,1::2],axis=-1)-np.sum(boxes1[...,1::2], axis=-1))/ha/2., np.zeros_like(ha))
    tw = np.where(wa>0, np.log((boxes2[...,2]-boxes2[...,0])/wa), np.zeros_like(wa))    # log(w/wa)
    th = np.where(ha>0, np.log((boxes2[...,3]-boxes2[...,1])/ha), np.zeros_like(ha))
    offset = np.stack([tx,ty,tw,th], axis=-1)    # [N1,N2,4]
    # print(np.max(offset))

    # # vis anchors
    # img = np.ones((600,800,3))
    # labels = np.ones((boxes1.shape[0],1))
    # boxes = np.concatenate([boxes1.reshape((-1,4)), labels], axis=-1)
    # xc = (boxes[...,0]+boxes[...,2])/2.
    # yc = (boxes[...,1]+boxes[...,3])/2.
    # img = draw_box(img, boxes)
    # img = draw_points(img, np.stack([xc,yc], axis=-1))
    # cv2.imshow("tmp", img)
    # cv2.waitKey(0)

    return iou, offset


def draw_box(img, boxes, line_size=2):
    abs_h, abs_w, c = img.shape
    img = img.copy()
    for b in boxes:
        if b.shape[0]==5:
            x1, y1, x2, y2, clsid = b
        else:
            x1, y1, x2, y2 = b
        abs_x1 = int(abs_w*x1)
        abs_y1 = int(abs_h*y1)
        abs_x2 = int(abs_w*x2)
        abs_y2 = int(abs_h*y2)
        cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), (255,0,0), line_size)
    return img


def draw_points(img, points):
    abs_h, abs_w, c = img.shape
    img = img.copy()
    for p in points:
        xc, yc = p
        abs_xc = int(abs_w*xc)
        abs_yc = int(abs_h*yc)
        cv2.circle(img, (abs_xc, abs_yc), 2, (0,0,255))
    return img


if __name__ == '__main__':

    img_dir = "data/"
    label_dir = "data/"
    num_classes = 2
    input_shape = (512,512,3)   # shorter 600
    batch_size = 1
    max_boxes = 20

    generator = dataSequence(img_dir, label_dir, num_classes, output_stride=16,
                             input_shape=input_shape, batch_size=batch_size,
                             max_boxes=max_boxes)

    for idx, data_batch in enumerate(generator):
        print('idx: ', idx)

        image_batch, gt_batch = data_batch[0]    # [b,h,w,3], [b,M,c+1+4]
        print(gt_batch.shape)
        for i in range(batch_size):
            img = image_batch[i]
            boxes = gt_batch[i,:,-4:]
            img = draw_box(img, boxes)
            cv2.imshow("tmp", img)
            cv2.waitKey(0)

        # if idx>100:
        #     break




    




