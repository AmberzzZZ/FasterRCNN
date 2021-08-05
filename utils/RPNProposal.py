from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf
import sys
sys.path.append("..")
from config import config


class RPNProposal(Layer):
    '''
    # inputs: rpn_raw_outputs
        rpn_objectness: [b,h,w,k]
        rpn_boxoffset: [b,h,w,4k]
        gt_clsoffset: [b,h,w,c+1+4]
    # outputs: proposal coords
        roi_proposals: [b,N,4], sorted by score, x1y1x2y2, normed
        roi_proposals_gt: [b,N,c+1+4c], one-hot label & offsets per cls
    '''

    def __init__(self, n_anchors=9, n_classes=80, output_stride=16, top_n=2000,
                 mode='train',
                 fg_iou_thresh=0.5, bg_iou_thresh=0.5, max_boxes=20,
                 positive_fraction=0.25, batch_size_per_img=512,
                 score_thresh=0.5,
                 **kwargs):
        super(RPNProposal, self).__init__(**kwargs)
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.output_stride = output_stride
        self.top_n = top_n
        self.mode = mode

        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.max_boxes = max_boxes
        self.positive_fraction = positive_fraction
        self.batch_size_per_img = batch_size_per_img

    def compute_output_shape(self, input_shape):
        if self.mode=='train':
            # train mode: return rois & roi targets
            # balance pos/neg per image
            return [(None, self.batch_size_per_img, 4), (None, self.batch_size_per_img, self.n_classes+1+4*self.n_classes)]
        else:
            # test mode: return topN proposals
            return (None, self.top_n, 4)

    def call(self, x):
        # sort and keep topK（2000）
        scores = x[0]   # [b,h,w,k,1]
        offsets = x[1]    # [b,h,w,k,4], pred txtytwth

        # shape
        batch = K.shape(scores)[0]
        hs = K.shape(scores)[1]    # feature map level: hs,ws
        ws = K.shape(scores)[2]

        # preds: offset2abs2norm
        anchors = K.reshape(K.variable(config.anchors), (1,1,1,self.n_anchors,2))  # [1,1,1,9,2] wh, origin level
        gridx, gridy = tf.meshgrid(tf.range(ws), tf.range(hs))
        center_xy = (tf.cast(tf.stack([gridx, gridy], axis=-1), 'float32')+0.5)*self.output_stride   # [h,w,2], origin level
        center_xy = K.expand_dims(center_xy, axis=-2)   # [h,w,1,2]
        wh = tf.cast(tf.stack([ws,hs]),dtype='float32') * self.output_stride
        pred_wh = K.exp(offsets[...,2:]) * anchors / wh   # pred, normed, wh, [b,h,w,a,2]
        pred_xcyc = (anchors * offsets[...,:2] + center_xy) / wh      # pred, normed, xcyc, [b,h,w,a,2]
        pred_x1y1 = pred_xcyc - pred_wh/2.
        pred_x2y2 = pred_xcyc + pred_wh/2.
        pred_x1y1x2y2c = K.concatenate([pred_x1y1,pred_x2y2,scores], axis=-1)
        # filter proposals(out of boundary & every small)
        pred_x1y1x2y2c = self.filter(pred_x1y1x2y2c)
        # sort and keep topK, sort by each sample, loop b
        pred_x1y1x2y2c = K.reshape(pred_x1y1x2y2c, (batch,-1,5))   # [b,N,5]
        def loop_body(b, proposals):
            res = tf.math.top_k(pred_x1y1x2y2c[b,...,-1], k=self.top_n)  # [N,5]
            boxes_ = tf.gather(pred_x1y1x2y2c[b], res.indices)[...,:4]   # [topK,4]
            # nms
            scores_ = tf.gather(pred_x1y1x2y2c[b], res.indices)[...,-1]  # [topK,]
            nms_indices = tf.image.non_max_suppression(boxes_, scores_, self.top_n, iou_threshold=0.7)
            rois_current_sample = tf.gather(boxes_, nms_indices)   # [N,4]
            pad = self.top_n-tf.shape(rois_current_sample)[0]
            rois_current_sample = tf.pad(rois_current_sample, [[0,pad], [0,0]])  # [topK,4]
            rois_current_sample = tf.reshape(rois_current_sample, (self.top_n,4))
            proposals = proposals.write(b, rois_current_sample)
            return b+1, proposals
        proposals = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        _, proposals = tf.while_loop(lambda b,*args: b<batch, loop_body, [0, proposals])
        proposals = proposals.stack()    # [b,N,4]

        if self.mode=='test':
            return proposals

        else:
            # compute targets
            gt_boxes = x[2][...,-4:]    # [b,N,4], gt normed x1y1x2y2
            gt_labels = x[2][...,:-4]    # [b,N,c+1], one-hot label

            iou = self.cal_iou(proposals, gt_boxes)    # [b,N,M]
            iou = tf.Print(iou, [K.max(iou)], message='max iou')
            matched_iou = tf.sort(iou, axis=-1, direction='DESCENDING')[...,0]   # [b,N,]
            matched_idx = tf.argsort(iou, axis=-1, direction='DESCENDING')[...,0]
            best_matched_iou = tf.sort(iou, axis=1, direction='DESCENDING')[:,0,:]   # [b,M,]
            best_matched_idx = tf.argsort(iou, axis=1, direction='DESCENDING')[:,0,:]   # [b,M,]

            def loop_body(b, rois, rois_gt):
                # random sampler
                positives = tf.where(matched_iou[b]>self.fg_iou_thresh)  # [n_pos,1]
                # add best match??
                gt_best_match = tf.cast(tf.gather(best_matched_idx[b], tf.where(best_matched_iou[b]>0)), 'int64')
                positives = K.concatenate([positives, gt_best_match], axis=0)
                negatives = tf.where(matched_iou[b]<self.bg_iou_thresh)  # [n_neg,1]
                n_pos = K.minimum(tf.size(positives), int(self.batch_size_per_img*self.positive_fraction))
                n_neg = 0  # K.minimum(self.batch_size_per_img - n_pos, tf.size(negatives))
                n_neg = tf.Print(n_neg, [n_pos, n_neg], message="n_pos & n_neg for detector per image")
                positives = tf.random_shuffle(positives)[:n_pos]
                negatives = tf.random_shuffle(negatives)[:n_neg]
                # encode pos targets
                pos_proposals = tf.gather(proposals[b], positives[:,0])   # [n_pos,4]
                matched_gt_indices = tf.gather(matched_idx[b], positives[:,0])    # [n_pos,]
                matched_gt_boxes = tf.gather(gt_boxes[b], matched_gt_indices)   # [n_pos, 4]
                pos_gt_offsets = self.encode_offset(matched_gt_boxes, pos_proposals)
                pos_gt_offsets = tf.tile(pos_gt_offsets, [1,self.n_classes])   # [n_pos, 4c]
                pos_gt_labels = tf.gather(gt_labels[b], matched_gt_indices)   # [n_pos, c+1]
                cls_mask = tf.reshape(tf.tile(K.expand_dims(pos_gt_labels[:,:self.n_classes],axis=-1), [1,1,4]), (K.shape(pos_gt_labels)[0],self.n_classes*4))  # [n_pos,4c]
                pos_gt_offsets = pos_gt_offsets * cls_mask
                # neg targets
                neg_proposals = tf.gather(proposals[b], negatives[:,0])   # [n_neg,4]
                neg_gt_offsets = tf.zeros((n_neg, self.n_classes*4))
                neg_gt_labels = tf.concat([tf.zeros((n_neg, self.n_classes)), tf.ones((n_neg, 1))], axis=-1)   # [n_neg, c+1]
                # gather
                proposals_b = tf.concat([pos_proposals, neg_proposals], axis=0)   # [N,4]
                proposals_gt_offsets_b = tf.concat([pos_gt_offsets, neg_gt_offsets], axis=0)   # [N,4c]
                proposals_gt_labels_b = tf.concat([pos_gt_labels, neg_gt_labels], axis=0)  # [N,c+1]
                proposals_gt_b = tf.concat([proposals_gt_labels_b, proposals_gt_offsets_b], axis=-1)            
                # pad if not enough
                pad = self.batch_size_per_img - tf.shape(proposals_b)[0]
                proposals_b = tf.pad(proposals_b, [[0,pad],[0,0]])     # [batch_per_img, 4]
                proposals_gt_b = tf.pad(proposals_gt_b, [[0,pad],[0,0]])     # [batch_per_img, c+1+4c]
                rois = rois.write(b, proposals_b)
                rois_gt = rois_gt.write(b, proposals_gt_b)
                return b+1, rois, rois_gt
            rois = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            rois_gt = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            _, rois, rois_gt = tf.while_loop(lambda b,*args: b<batch, loop_body, [0, rois, rois_gt])
            rois = rois.stack()
            rois_gt = rois_gt.stack()

            return [rois, rois_gt]

    def encode_offset(self, gt_boxes, ref_boxes):
        # encode gt offsets on proposals
        # boxes: [N,4], normed, x1y1x2y2, gt/proposals
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

    def filter(self, pred_x1y1x2y2c, min_size=1e-2):
        # pred_x1y1x2y2c: [b,hs,ws,a,4+1], normed
        # clip boundary
        pred_x1y1x2y2c = K.clip(pred_x1y1x2y2c, min_value=0., max_value=1.)
        # mute small (1e-2)
        pred_wh = pred_x1y1x2y2c[...,2:4] - pred_x1y1x2y2c[...,0:2]
        valid_mask = tf.where((pred_wh[...,0]>min_size) & (pred_wh[...,1]>min_size), tf.ones_like(pred_wh[...,0]), tf.zeros_like(pred_wh[...,0]))
        valid_mask = K.expand_dims(valid_mask, axis=-1)
        return pred_x1y1x2y2c * valid_mask


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

    x1 = Input((32,32,9,1))  # [b,h,w,a,1]
    x2 = Input((32,32,9,4))  # [b,h,w,a,4]
    gt = Input((3,20+1+4))  # [b,h,w,a,c+1+4]
    y = RPNProposal(n_anchors=9, n_classes=20, mode='train')([x1,x2,gt])
    print(y)   # [b,2000,4] & [b,2000,20+1+4*20]
    model = Model([x1, x2, gt], y)

    import numpy as np
    x1 = np.random.uniform(size=(5,32,32,9,1))
    x2 = np.random.uniform(size=(5,32,32,9,4))
    gt = np.random.uniform(size=(5,3,20+1+4))
    if 1:
        outputs = model.predict([x1, x2, gt])
        print(outputs[0].shape, outputs[1].shape)   # [b,2000,4] & [b,2000,20+1+4*20]
    else:
        output = model.predict([x1, x2, gt])
        print(output.shape)






