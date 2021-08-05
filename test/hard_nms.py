# hard nms
import numpy as np
import cv2


def hard_nms(boxes, scores, labels, score_thresh=0.1, iou_thresh=0.3, max_detections=200):
    boxes = boxes.copy()
    scores = scores.copy()
    labels = labels.copy()
    # boxes: [N,4], x1y1x2y2
    # scores: [N,1]
    # labels: [N,1]

    # score filter
    valid_indices = np.where(scores>score_thresh)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]

    # sort decending
    indices = np.argsort(scores, axis=0)[::-1,0]
    indices = indices[:max_detections]

    # cal_iou
    iou = cal_iou(boxes[...,:4], boxes[...,:4])     # (N2,N1)

    # tranverse each high score box
    picked = []
    while indices.shape[0]:
        current = indices[0]
        picked.append(current)
        if indices.shape[0]==1:
            break
        indices = indices[1:]
        current_iou = iou[current][indices]
        indices = indices[current_iou<iou_thresh]
    # picked = np.stack(picked, axis=-1)

    return boxes[picked], scores[picked], labels[picked]


def cal_iou(boxes1, boxes2, epsilon=1e-5):
    boxes1 = boxes1.copy()
    boxes2 = boxes2.copy()
    # boxes1: [N1,4], x1y1x2y2
    # boxes2: [N2,4], x1y1x2y2

    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    inter_mines = np.maximum(boxes1[...,:2], boxes2[...,:2])    # [N1,N2,2]
    inter_maxes = np.minimum(boxes1[...,2:], boxes2[...,2:])
    inter_wh = np.maximum(inter_maxes - inter_mines, 0.)
    inter_area = inter_wh[...,0] * inter_wh[...,1]

    box_area1 = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
    box_area1 = np.tile(box_area1, [1,np.shape(boxes2)[0]])
    box_area2 = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])
    box_area2 = np.tile(box_area2, [np.shape(boxes1)[0],1])

    iou = inter_area / (box_area1 + box_area2 - inter_area + epsilon)

    return iou


if __name__ == '__main__':

    bboxes = np.array([[30,12,50,28,0.9],
                       [25,10,52,25,0.95],
                       [27,11,55,23,0.8],
                       [33,9,44,26,0.6],
                       [80,88,100,122,0.9]])
    box, score, _ = hard_nms(bboxes[...,:4], bboxes[...,-1:], bboxes[...,-1:],
                             iou_thresh=0.3, score_thresh=0.5)

    print("result: ")
    print(box)
    print(score)

    # vis
    canvas = np.zeros((150, 150))
    cv2.rectangle(canvas, (30,12), (50,28), 255, 1)
    cv2.rectangle(canvas, (25,10), (52,25), 255, 1)
    cv2.rectangle(canvas, (27,11), (55,23), 255, 1)
    cv2.rectangle(canvas, (33,9), (44,26), 255, 1)
    cv2.rectangle(canvas, (80,88), (100,122), 255, 1)
    cv2.imshow("before nms", canvas)
    cv2.waitKey(0)

    canvas = np.zeros((150, 150))
    for b in box:
        cv2.rectangle(canvas, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), 255, 1)
    cv2.imshow("after nms", canvas)
    cv2.waitKey(0)








