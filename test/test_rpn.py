import cv2
import numpy as np
import pandas as pd
import os
from hard_nms import hard_nms, cal_iou
import sys
sys.path.append("..")
from fasterRCNN import fasterRCNN
from dataSequence import category_name, get_box
from config import config


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


if __name__ == '__main__':

    input_shape = (512,512,3)
    n_classes = len(category_name)
    output_stride = 16
    anchors = config.anchors
    n_anchors = config.n_anchors

    # model
    rpn_model, detection_model = fasterRCNN(input_shape, n_classes, n_anchors, mode='test')
    rpn_model.load_weights("rpn.h5")

    test_lst = ['tux_positive']
    data_dir = "../data/"
    for file in test_lst:
        img = cv2.imread(os.path.join(data_dir, file+'.jpg'),1)
        img = cv2.resize(img, input_shape[:2])
        inpt = np.expand_dims(img/255., axis=0)   # [1,512,512,3]
        rpn_objectness, rpn_boxoffset, rois = rpn_model.predict(inpt)   # [b,hs,ws,a,1], [b,hs,ws,a,4], [b,topN,4]

        ######## vis test-time proposals
        img1 = img.copy()
        img1 = draw_box(img1, rois, line_size=2)
        cv2.imshow("proposals", img1)
        cv2.waitKey(0)
        ######## assess proposals
        gt_boxes = get_box(os.path.join(data_dir, file+'.json'))
        rois = rois[0]
        n_proposals = rois.shape[0]
        rois, _, _ = hard_nms(rois, np.ones((n_proposals,1)), np.ones((n_proposals,1)), iou_thresh=0.7)
        iou = cal_iou(rois, gt_boxes)   # [N,M]
        best_iou = np..max(iou, axis=1)  # [N,1]
        good_proposals = np.where(best_iou>0.5)
        print("good proposals: ", len(good_proposals[0]), ' /among: ', n_proposals)

        ######### rpn prediction
        hm = rpn_objectness[0]
        fg_coords = np.where(hm>0.5)
        box_preds = rpn_boxoffset[0][fg_coords]    # [n_fg,4], txtytwth

        ws, hs = input_shape[1]//output_stride, input_shape[0]//output_stride
        w, h = ws*output_stride, hs*output_stride
        grid_x, grid_y = np.meshgrid(np.arange(ws), np.arange(hs))
        grid_xy = np.stack([grid_x, grid_y], axis=-1)
        anchors_xcyc = output_stride * (np.expand_dims(grid_xy, axis=-2)+0.5)   # [h,w,1,2]
        anchors_xcyc = np.tile(anchors_xcyc, [1,1,n_anchors,1])/np.array([w,h])  # [h,w,a,2], normed
        anchors_wh = np.tile(np.reshape(anchors, (1,1,n_anchors,2)), [hs,ws,1,1])/np.array([w,h])

        pred_xcyc = anchors_wh[fg_coords] * box_preds[:,:2] + anchors_xcyc[fg_coords]
        pred_wh = anchors_wh[fg_coords] * np.exp(box_preds[:,2:])
        pred_x1y1 = pred_xcyc - pred_wh/2.
        pred_x2y2 = pred_xcyc + pred_wh/2.

        ######## vis rpn prediction
        pred_boxes = np.concatenate([pred_x1y1,pred_x2y2],axis=-1)   # [n_fg,4], normed, x1y1x2y2
        img2 = img.copy()
        img2 = draw_box(img2, pred_boxes, line_size=2)
        cv2.imshow("proposals", img2)
        cv2.waitKey(0)







