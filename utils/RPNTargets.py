from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("..")
from config import config


class RPNTargets(Layer):
    '''
    # inputs: gt boxes, [b,M,c+1+4]
    # outputs: rpn targets, [b,h,w,a,1+4]
        rpn_obj_targets: [b,h,w,a,1], objectness, [-1,0,1], random 256 balance pos/neg samples
        rpn_box_targets: [b,h,w,a,4], gt_anchor_offsets, tx_ty_tw_th
    '''

    def __init__(self, input_hw, output_stride=16,
                 fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 positive_fraction=0.5, batch_size_per_img=256,
                 **kwargs):
        super(RPNTargets, self).__init__(**kwargs)
        self.input_hw = input_hw
        self.output_stride = output_stride
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.positive_fraction = positive_fraction
        self.batch_size_per_img = batch_size_per_img

        self.n_anchors = config.n_anchors
        self.anchors = config.anchors
        self.hs = input_hw[0] // output_stride
        self.ws = input_hw[1] // output_stride

        # anchors: abs2norm
        anchors = K.reshape(K.variable(config.anchors), (1,1,1,self.n_anchors,2))  # [1,1,1,9,2] wh, origin level
        gridx, gridy = tf.meshgrid(tf.range(self.ws), tf.range(self.hs))
        center_xy = (tf.cast(tf.stack([gridx, gridy], axis=-1), 'float32')+0.5)*self.output_stride   # [h,w,2], origin level
        center_xy = K.expand_dims(center_xy, axis=-2)   # [h,w,1,2]
        wh = tf.cast(tf.stack([self.ws,self.hs]),dtype='float32') * self.output_stride
        anchors_x1y1 = (center_xy - anchors/2.)/wh    # normed, x1y1, [1,h,w,a,2]
        anchors_x2y2 = (center_xy + anchors/2.)/wh
        # filter anchors(out of boundary)
        self.anchors_x1y1x2y2 = self.filter(anchors_x1y1, anchors_x2y2)

    def compute_output_shape(self, input_shape):
        return (None, self.hs, self.ws, self.n_anchors, 1+4)

    def call(self, x):
        gt_boxes = x[...,-4:]    # [b,M,4], gt normed x1y1x2y2

        # shape
        batch = K.shape(gt_boxes)[0]
        max_boxes = K.shape(gt_boxes)[1]
        ws = self.ws
        hs = self.hs
        n_anchors = self.n_anchors

        N = hs*ws*n_anchors
        anchors_x1y1x2y2 = K.reshape(self.anchors_x1y1x2y2, (1,N,4))   # [1,hwa,4]

        # compute targets
        iou = self.cal_iou(anchors_x1y1x2y2, gt_boxes)    # [b,N,M]
        matched_iou = tf.sort(iou, axis=-1, direction='DESCENDING')[...,0]   # [b,N,]
        matched_idx = tf.argsort(iou, axis=-1, direction='DESCENDING')[...,0]
        best_match_iou = tf.sort(iou, axis=1, direction='DESCENDING')[:,0,:]   # [b,M,]
        best_match = tf.argsort(iou, axis=1, direction='DESCENDING')[:,0,:]

        def loop_body(b, rpn_targets):
            # random sampler
            positives = tf.where(matched_iou[b]>self.fg_iou_thresh)  # [n_pos,1]
            gt_best_match = tf.cast(tf.gather(best_match[b], tf.where(best_match_iou[b]>0)), 'int64') # [valid_gt,1]
            positives = K.concatenate([positives, gt_best_match], axis=0)
            negatives = tf.where(matched_iou[b]<self.bg_iou_thresh)  # [n_neg,1]
            n_pos = tf.size(positives)
            n_pos = K.minimum(n_pos, int(self.batch_size_per_img*self.positive_fraction))
            n_neg = self.batch_size_per_img - n_pos
            n_neg = tf.Print(n_neg, [n_pos, n_neg], message="n_pos & n_neg for RPN per image")
            positives = tf.random_shuffle(positives)[:n_pos]
            negatives = tf.random_shuffle(negatives)[:n_neg]
            pos_mask = tf.scatter_nd(positives, tf.ones((n_pos,)), (N,))
            neg_mask = tf.scatter_nd(negatives, tf.ones((n_neg,)), (N,))
            # encode pos targets
            pos_anchors = tf.gather(anchors_x1y1x2y2[0], positives[:,0])   # [n_pos,4]
            matched_gt_indices = tf.gather(matched_idx[b], positives[:,0])    # [n_pos,]
            matched_gt_boxes = tf.gather(gt_boxes[b], matched_gt_indices)   # [n_pos, 4]
            pos_gt_offsets = self.encode_offset(matched_gt_boxes, pos_anchors)  # [n_pos, 4]
            # pos_gt_offsets = tf.Print(pos_gt_offsets, [positives, pos_gt_offsets[0]], message='before', summarize=200)
            pos_gt_offsets = tf.scatter_nd(positives, pos_gt_offsets, (N,4))  # [N,4]
            # add cls targets [0,1,-1]
            cls_targets = -tf.ones((N))
            cls_targets = tf.where(pos_mask>0, tf.ones_like(cls_targets), cls_targets)
            cls_targets = tf.where(neg_mask>0, tf.zeros_like(cls_targets), cls_targets)
            cls_targets = tf.expand_dims(cls_targets, axis=-1)
            # rpn targets
            rpn_target = K.concatenate([cls_targets,pos_gt_offsets], axis=-1)
            # rpn_target = tf.Print(rpn_target, [rpn_target[positives[0,0]]], message='after', summarize=200)
            rpn_targets = rpn_targets.write(b, rpn_target)   # [N,1]
            return b+1, rpn_targets

        rpn_targets = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        _, rpn_targets = tf.while_loop(lambda b,*args: b<batch, loop_body, [0, rpn_targets])
        rpn_targets = rpn_targets.stack()
        rpn_targets = tf.reshape(rpn_targets, (batch, hs, ws, n_anchors, 1+4))

        return rpn_targets

    def encode_offset(self, gt_boxes, ref_boxes):
        # encode gt offsets on anchors
        # boxes: [N,4], normed, x1y1x2y2, gt/anchor
        w_gt = gt_boxes[:,2] - gt_boxes[:,0]
        h_gt = gt_boxes[:,3] - gt_boxes[:,1]
        xc_gt = (gt_boxes[:,2] + gt_boxes[:,0])/2
        yc_gt = (gt_boxes[:,3] + gt_boxes[:,1])/2

        w_ref = ref_boxes[:,2] - ref_boxes[:,0]
        h_ref = ref_boxes[:,3] - ref_boxes[:,1]
        xc_ref = (ref_boxes[:,2] + ref_boxes[:,0])/2
        yc_ref = (ref_boxes[:,3] + ref_boxes[:,1])/2

        # tx: (x-x_gt)/w_gt
        tx = tf.where(w_gt>0, (xc_gt-xc_ref) / w_ref, tf.zeros_like(w_ref))
        ty = tf.where(h_gt>0, (yc_gt-yc_ref) / h_ref, tf.zeros_like(h_ref))

        # tw: log(w/w_gt)
        tw = tf.where(w_gt>0, tf.log(w_gt/w_ref), tf.zeros_like(w_ref))
        th = tf.where(h_gt>0, tf.log(h_gt/h_ref), tf.zeros_like(h_ref))

        offsets = tf.stack([tx,ty,tw,th], axis=-1)    # [N,4]

        return offsets

    def filter(self, anchors_x1y1, anchors_x2y2):
        # filter anchors with edges out of boundary
        # anchors_x1y1: [1,h,w,a,2], normed
        # anchors_x2y2: [1,h,w,a,2]
        valid_mask = tf.where(tf.cast(anchors_x1y1[...,0]>0,'bool') &
                              tf.cast(anchors_x1y1[...,1]>0,'bool') &
                              tf.cast(anchors_x2y2[...,0]<1,'bool') &
                              tf.cast(anchors_x2y2[...,1]<1,'bool'),
                              tf.ones_like(anchors_x1y1[...,0]),
                              tf.zeros_like(anchors_x1y1[...,0]))
        valid_mask = tf.expand_dims(valid_mask, axis=-1)
        anchors_x1y1x2y2 = K.concatenate([anchors_x1y1, anchors_x2y2], axis=-1)
        return anchors_x1y1x2y2 * valid_mask

    def cal_iou(self, proposal_boxes, gt_boxes):
        proposal_boxes = tf.expand_dims(proposal_boxes, axis=2)   # [b,N1,1,4]
        gt_boxes = tf.expand_dims(gt_boxes, axis=1)   # [b,1,N2,4]

        inter_mines = tf.maximum(proposal_boxes[...,:2], gt_boxes[...,:2])    # [b,N1,N2,2]
        inter_maxes = tf.minimum(proposal_boxes[...,2:], gt_boxes[...,2:])
        inter_wh = tf.maximum(inter_maxes - inter_mines, 0.)
        inter_area = inter_wh[...,0] * inter_wh[...,1]    # [b,N1,N2]

        box_area1 = (proposal_boxes[...,2]-proposal_boxes[...,0]) * (proposal_boxes[...,3]-proposal_boxes[...,1])   # [b,N1,1]
        box_area2 = (gt_boxes[...,2]-gt_boxes[...,0]) * (gt_boxes[...,3]-gt_boxes[...,1])   # [b,1,N2]

        iou = inter_area / (box_area1 + box_area2 - inter_area + K.epsilon())   # [b,N1,N2]

        return iou


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model

    x = Input((None,20+1+4))  # [b,M,c+1+4]
    y = RPNTargets(input_hw=(512,512,3))(x)
    print('model graph output: ', y)   # [b,h,w,a,1+4]
    model = Model(x, y)

    import numpy as np
    gt = np.random.uniform(size=(5,3,20+1+4))
    output = model.predict(gt)
    print('model running output: ', output.shape)      # [b,h,w,a,1+4]






