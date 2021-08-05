import keras.backend as K
import tensorflow as tf
from config import config


anchors = config.anchors   # [9,2]
n_anchors = anchors.shape[0]


def rpnet_loss(args):
    # cls_output: [b,N,1]
    # box_output: [b,N,4]
    # roi_targets: [b,N,1+4]
    cls_output, box_output, roi_targets = args

    loss_cls = rpn_loss_cls(roi_targets[..., :1], cls_output)
    loss_box = rpn_loss_box(roi_targets, box_output)

    return loss_cls + loss_box


def detection_loss(args, n_classes):
    # cls_output: [b,N,c+1]
    # box_output: [b,N,4]
    # roi_targets: [b,N,c+1+4c]
    cls_output, box_output, roi_targets = args
    cls_gt = roi_targets[..., :n_classes+1]
    box_gt = roi_targets[..., n_classes+1:]

    loss_cls = detection_loss_cls(cls_gt, cls_output)
    # fg mask
    # n_pos = K.sum(cls_gt[...,:n_classes])
    fg_mask = tf.tile(K.expand_dims(cls_gt[...,:n_classes], axis=-1), [1,1,1,4])  # [b,N,c,4]
    fg_mask = tf.reshape(fg_mask, (K.shape(fg_mask)[0], K.shape(fg_mask)[1], n_classes*4))
    loss_box = detection_loss_box(box_gt, box_output, n_classes, fg_mask)

    return loss_cls + loss_box


def rpn_loss_cls(y_true, y_pred):
    # bce, compute on random 256 balanced pos/neg samples
    # y_true: [b,h,w,k,1], per location, per anchor, [0,1,-1]
    # y_pred: [b,h,w,k,1]

    # bce:
    pt = K.clip(K.abs(y_true-y_pred), K.epsilon(), 1-K.epsilon())   # gap
    bce = -K.log(1-pt)
    # valid mask
    valid_mask = tf.where(y_true>-1, tf.ones_like(y_pred), tf.zeros_like(y_pred))
    # norm
    N_cls = 256.
    return K.sum(bce*valid_mask) / N_cls


def rpn_loss_box(y_true, y_pred):
    # smooth L1, compute on picked pos samples
    # y_true: [b,h,w,k,5], ob+txtytwth
    # y_pred: [b,h,w,k,4]

    # filter proposals in torch vision
    pass

    # smooth L1:
    pt = K.abs(y_true[...,1:]-y_pred)
    smooth_l1 = tf.where(pt<1, 0.5*pt*pt, pt-0.5)    # [b,h,w,k,4]
    # valid mask
    valid_mask = tf.where(y_true[...,0]>0, tf.ones_like(y_true[...,0]), tf.zeros_like(y_true[...,0]))   # [b,h,w,k]
    valid_mask = K.tile(K.expand_dims(valid_mask,axis=-1), [1,1,1,1,4])   # [b,h,w,k,4]
    # norm
    N_reg = 2400.
    return K.sum(smooth_l1*valid_mask) / N_reg


def detection_loss_cls(y_true, y_pred):
    # ce, compute on topk(2000) proposals
    # y_true: [b,topK,c+1], per image, per proposal
    # y_pred: [b,topK,c+1]

    # ce:
    pt = K.clip(K.abs(y_true-y_pred), K.epsilon(), 1-K.epsilon())
    ce = -K.log(1-pt) * y_true
    # norm factor: topK?
    N = K.cast(K.shape(y_pred)[1], 'float32')
    return K.sum(ce) / N


def detection_loss_box(y_true, y_pred, n_classes, fg_mask):
    # smooth L1, compute on topk(2000) proposals
    # y_true: [b,topK,4c], per image, per proposal
    # y_pred: [b,topK,4c]
    # fg_mask: [b,topK,4c]

    # smooth L1:
    pt = K.abs(y_true-y_pred)
    smooth_l1 = tf.where(pt<1, 0.5*pt*pt, pt-0.5)    # [b,topK,4c]
    # balance factor: [10,10,5,5] in torch vision
    balance_factor = tf.constant([10,10,5,5], dtype='float32')
    balance_factor = tf.tile(balance_factor, [n_classes])   # [4c]
    # norm factor: n_pos?
    N = K.sum(fg_mask)
    return K.sum(smooth_l1*balance_factor*fg_mask) / N


def constant_loss(y_true, y_pred):
    # for the frozen branch
    return K.constant(0.)




