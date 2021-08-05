from keras.layers import Input
from keras.models import Model
from RPNTargets import RPNTargets
from RPNProposal import RPNProposal
from ROIAlign import RoiAlign
from ROIPooling import RoiPooling
import numpy as np
import json
import cv2
import sys
sys.path.append("..")
from config import config


def get_box(yolo_file):
    label_dict = {'cat': 0, 'dog': 1}
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


if __name__ == '__main__':

    tests = {'RPNTargets':0,
             'RPNProposal':0,
             'ROIAlign':0,
             'RoiPooling':1,
    }

    if tests['RPNTargets']:
        ############ test RPNTargets ############
        img = cv2.imread("../data/tux_positive.jpg", 1)
        img = cv2.resize(img, (512,512))
        boxes = get_box("../data/tux_positive.json")
        labels = boxes[:,-1]
        boxes = boxes[:,:-1]
        n_classes = 2
        n_boxes = boxes.shape[0]
        max_boxes = 10
        gt = np.zeros((1,max_boxes, n_classes+1+4))
        indices = np.arange(n_boxes)
        if n_boxes>max_boxes:
            np.random.shuffle(indices)
            indices = indices[:max_boxes]
        for idx, box_id in enumerate(indices):
            cls_id = int(labels[box_id])
            gt[0,idx,cls_id] = 1
            gt[0,idx,-4:] = boxes[box_id]

        # model
        x = Input((None,n_classes+1+4))  # [b,M,c+1+4]
        y = RPNTargets(input_hw=(512,512,3))(x)  # [b,h,w,a,1+4]
        model = Model(x, y)

        gt_offsets = model.predict(gt)[0]   # [h,w,a,1+4], offsets, txtytwth
        coords = np.where(gt_offsets[...,0]>0)
        print("n_pos", len(coords[0]), gt_offsets.shape)
        print("pos targets", gt_offsets[coords])    # [N,1+4]
        anchors = config.anchors
        n_anchors = config.n_anchors
        hs, ws = 512//16, 512//16
        gridx, gridy = np.meshgrid(np.arange(ws), np.arange(hs))
        center_xy = (np.stack([gridx, gridy], axis=-1)+0.5)*16   # [h,w,2], origin level xcyc
        anchors_xcyc = np.tile(np.expand_dims(center_xy, axis=-2), [1,1,n_anchors,1])  # [h,w,a,2]
        anchors_wh = np.tile(anchors, [hs,ws,1,1])  # [h,w,a,2]
        anchors_xcyc = anchors_xcyc[coords] / np.array([512,512])
        anchors_wh = anchors_wh[coords] / np.array([512,512])        # normed
        gt_offsets = gt_offsets[coords]
        pred_xcyc = gt_offsets[:,1:3] * anchors_wh + anchors_xcyc
        pred_wh = np.exp(gt_offsets[:,3:]) * anchors_wh
        pred_x1y1 = pred_xcyc - pred_wh/2.
        pred_x2y2 = pred_xcyc + pred_wh/2.
        img = draw_box(img, np.concatenate([pred_x1y1,pred_x2y2], axis=-1), line_size=2)
        cv2.imshow("tmp", img)
        cv2.waitKey(0)

        coords = np.where(gt_offsets[...,0]==0)
        offsets = gt_offsets[coords]
        print("n_neg", len(coords[0]))
        coords = np.where(gt_offsets[...,0]<0)
        print("ignore", len(coords[0]))

    if tests['RPNProposal']:
        ############ test RPNProposal ############
        img = cv2.imread("../data/tux_positive.jpg", 1)
        img = cv2.resize(img, (512,512))
        boxes = get_box("../data/tux_positive.json")
        labels = boxes[:,-1]
        boxes = boxes[:,:-1]
        n_classes = 2
        n_boxes = boxes.shape[0]
        max_boxes = 10
        gt = np.zeros((1,max_boxes, n_classes+1+4))
        indices = np.arange(n_boxes)
        if n_boxes>max_boxes:
            np.random.shuffle(indices)
            indices = indices[:max_boxes]
        for idx, box_id in enumerate(indices):
            cls_id = int(labels[box_id])
            gt[0,idx,cls_id] = 1
            gt[0,idx,-4:] = boxes[box_id]
            # add padding bg label
        gt_bg = np.sum(gt[...,:n_classes], axis=-1)
        gt_bg = np.float32(gt_bg==0)
        gt[...,n_classes] = gt_bg

        x1 = np.random.uniform(size=(1,32,32,9,1))
        x2 = np.random.uniform(size=(1,32,32,9,4))

        # model
        rpn_obj = Input((32,32,9,1))  # [b,h,w,a,1]
        rpn_off = Input((32,32,9,4))  # [b,h,w,a,4]
        gt_boxes = Input((None,n_classes+1+4))  # [b,h,w,a,c+1+4]
        y = RPNProposal(n_anchors=9, n_classes=n_classes, mode='train',
                        top_n=200, batch_size_per_img=50)([rpn_obj,rpn_off,gt_boxes])
        model = Model([rpn_obj,rpn_off,gt_boxes], y)

        proposals, rois_targets = model.predict([x1,x2,gt])
        proposals = proposals[0]    # [N,4]
        print(proposals.shape)
        # img = draw_box(img, proposals)
        # cv2.imshow("tmp", img)
        # cv2.waitKey(0)

        rois_targets = rois_targets[0]     # [N,c+1+4c]
        print(rois_targets[...,:n_classes].shape)
        rois_cls_targets = rois_targets[...,:n_classes].reshape((-1,n_classes))   # [N,c,]
        rois_box_targets = rois_targets[...,n_classes+1:].reshape((-1,n_classes,4))   # [N,c,4]
        rois_box_targets = rois_box_targets[rois_cls_targets>0]   # [n_pos,4], offsets, txtytwth
        rois_fg_targets = np.sum(rois_cls_targets, axis=-1)
        rois_raw_preds = proposals[rois_fg_targets>0]  # [n_pos,4], normed, x1y1x2y2, proposals
        xcyc = (rois_raw_preds[:,:2] + rois_raw_preds[:,2:]) / 2.
        wh = rois_raw_preds[:,2:] - rois_raw_preds[:,:2]
        pred_xcyc = rois_box_targets[:,:2] * wh + xcyc
        pred_wh = np.exp(rois_box_targets[:,2:]) * wh
        pred_x1y1 = pred_xcyc - pred_wh/2.
        pred_x2y2 = pred_xcyc + pred_wh/2.
        # img = draw_box(img, np.concatenate([pred_x1y1,pred_x2y2], axis=-1), line_size=2)
        # cv2.imshow("tmp", img)
        # cv2.waitKey(0)

    if tests['ROIAlign']:
        ############ test ROIAlign ############
        boxes = get_box("../data/tux_positive.json")
        boxes = np.expand_dims(boxes[:,:4],0)
        boxes = np.concatenate([boxes, np.array([[[0,0,1,1]]])], axis=1)
        img = cv2.imread("../data/tux_positive.jpg", 1)
        img = cv2.resize(img, (512,512))
        img = img.reshape((1,512,512,3))

        # model
        feature = Input((512,512,3))
        rois = Input((None,4))
        outputs = RoiAlign(pool_size=7)([feature, rois])
        model = Model([feature, rois], outputs)

        preds = model.predict([img, boxes])[0]
        for poolingmap in preds:
            print(poolingmap.shape)   # 'f32'
            tmp = cv2.resize(poolingmap, (256,256))
            cv2.imshow("tmp", tmp/255.)
            cv2.waitKey(0)

    if tests['RoiPooling']:
        ############ test RoiPooling ############
        boxes = get_box("../data/tux_positive.json")
        boxes = np.expand_dims(boxes[:,:4],0)
        boxes = np.concatenate([boxes, np.array([[[0,0,1,1]]])], axis=1)
        img = cv2.imread("../data/tux_positive.jpg", 1)
        img = cv2.resize(img, (512,512))
        img = img.reshape((1,512,512,3))

        # model
        feature = Input((512,512,3))
        rois = Input((3,4))
        outputs = RoiPooling(num_rois=3, pool_size=14)([feature, rois])
        model = Model([feature, rois], outputs)

        preds = model.predict([img, boxes])[0]
        for poolingmap in preds:
            print(poolingmap.shape)   # 'f32'
            tmp = cv2.resize(poolingmap, (256,256))
            cv2.imshow("tmp", tmp/255.)
            cv2.waitKey(0)









