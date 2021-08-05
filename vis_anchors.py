import cv2
import numpy as np
from config import config


def draw_box(img, boxes, line_size=2):
    abs_h, abs_w, c = img.shape
    img = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = b
        abs_x1 = int(abs_w*x1)
        abs_y1 = int(abs_h*y1)
        abs_x2 = int(abs_w*x2)
        abs_y2 = int(abs_h*y2)
        cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), (255,0,0), line_size)
    return img


def draw_points(img, points, radius=2, thickness=1):
    img = img.copy()
    for p in points:
        xc, yc = p
        abs_xc = int(xc)
        abs_yc = int(yc)
        cv2.circle(img, (abs_xc, abs_yc), radius, (0,0,255), thickness)
    return img


anchors = config.anchors
n_anchors = config.n_anchors
output_stride = 16
h, w = 600, 800
hs, ws = h//output_stride, w//output_stride
h, w = hs*output_stride, ws*output_stride


if __name__ == '__main__':

    grid_x, grid_y = np.meshgrid(np.arange(ws), np.arange(hs))
    grid_xy = np.stack([grid_x,grid_y], axis=-1)   # [h,w,2]
    center_xy = output_stride*(np.expand_dims(grid_xy, axis=-2) + 0.5)    # [h,w,1,2]
    anchors_x1y1 = center_xy - anchors/2.    # [h,w,a,2], abs
    anchors_x2y2 = center_xy + anchors/2.
    anchors_all = np.concatenate([anchors_x1y1, anchors_x2y2], axis=-1)
    # filter
    anchors_all *= (anchors_all[...,0:1]>0).astype(np.float32)
    anchors_all *= (anchors_all[...,1:2]>0).astype(np.float32)
    anchors_all *= (anchors_all[...,2:3]<ws*output_stride).astype(np.float32)
    anchors_all *= (anchors_all[...,3:4]<hs*output_stride).astype(np.float32)
    # norm
    anchors_all[...,0::2] /= output_stride*ws
    anchors_all[...,1::2] /= output_stride*hs    # [hs,ws,a,4], normed x1y1x2y2
    # vis static
    img = np.ones((h,w,3))
    for i in range(hs):
        for j in range(ws):
            anchor_boxes = anchors_all[i,j]   # [a,4]
            if np.sum(anchor_boxes):
                img = draw_points(img, [[(j+0.5)*output_stride, (i+0.5)*output_stride]])
    cv2.imshow("tmp", img)
    cv2.waitKey(0)
    # vis dynamic
    canvas = img.copy()
    for i in range(hs):
        for j in range(ws):
            # img = np.ones((h,w,3))
            img = canvas.copy()
            anchor_boxes = anchors_all[i,j]   # [a,4]
            if np.sum(anchor_boxes):
                img = draw_box(img, anchor_boxes, line_size=2)
                img = draw_points(img, [[(j+0.5)*output_stride, (i+0.5)*output_stride]], radius=3, thickness=-1)
                cv2.imshow("tmp", img)
                cv2.waitKey(3)




