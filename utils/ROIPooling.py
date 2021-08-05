from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class RoiPooling(Layer):
    '''
    # inputs
        feature map: [b,hs,ws,c]
        rpn pred: [b,N,4], normed, x1y1x2y2
    # outputs
        pooling maps: [b,topK,7,7,c]
    '''

    def __init__(self, num_rois, pool_size=7, pool_stride=16, **kwargs):

        super(RoiPooling, self).__init__(**kwargs)
        self.num_rois = num_rois
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def build(self, input_shape):
        self.n_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return (None, self.num_rois, self.pool_size, self.pool_size, self.n_channels)

    def call(self, x, mask=None):

        assert(len(x) == 2)

        feature_map = x[0]    # [b,h,w,c]
        rpn_pred = x[1]       # [b,N,4]

        # loop b
        def loop_body(b, roi_feature):
            crops = self.crop_and_pooling(feature_map[b], rpn_pred[b])   # [N,7,7,c]
            roi_feature = roi_feature.write(b, crops)
            return b+1, roi_feature
        batch = K.shape(feature_map)[0]     # batch size
        roi_feature = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        _, roi_feature = tf.while_loop(lambda b,*args: b<batch, loop_body, [0, roi_feature])
        roi_feature = roi_feature.stack()    # [b,N,7,7,c]

        return roi_feature

    def crop_and_pooling(self, feature_map, rpn_pred):
        # for a single image
        # featuremap: [hs,ws,c]
        # rpn_pred: [N,4]
        # returns: pooled proposal maps, [N,7,7,c]

        # pred on feature map, first approximation for rounding pixels
        hs = K.cast(K.shape(feature_map)[0], 'float32')
        ws = K.cast(K.shape(feature_map)[1], 'float32')
        x1 = K.cast(ws*rpn_pred[...,0], 'int32')
        y1 = K.cast(hs*rpn_pred[...,1], 'int32')
        x2 = K.cast(ws*rpn_pred[...,2], 'int32')
        y2 = K.cast(hs*rpn_pred[...,3], 'int32')
        # second approximation on pooling for abandoning boundaries
        stride_w = K.cast((x2-x1)/self.pool_size, 'int32')
        stride_h = K.cast((y2-y1)/self.pool_size, 'int32')
        x2 = x1 + stride_w * self.pool_size
        y2 = y1 + stride_h * self.pool_size

        # crop bboxes
        outputs = []
        for roi_idx in range(self.num_rois):
            bbox = feature_map[y1[roi_idx]:y2[roi_idx], x1[roi_idx]:x2[roi_idx]]
            bbox = K.reshape(bbox, (self.pool_size, stride_h[roi_idx], self.pool_size, stride_w[roi_idx], self.n_channels))
            bbox = tf.transpose(bbox, perm=(0,2,1,3,4))   # [7,7,binh,binw,C]
            bbox = tf.reshape(bbox, (self.pool_size, self.pool_size, -1, self.n_channels))
            bbox = K.max(bbox, axis=(2))   # [7,7,C]
            outputs.append(bbox)
        final_output = K.stack(outputs, axis=0)     # [300,7,7,c]

        return final_output


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model

    num_rois = 1000
    feature = Input((32,32,128))
    rois = Input((num_rois,4))
    outputs = RoiPooling(num_rois=num_rois)([feature, rois])
    print(outputs)
    model = Model([feature, rois], outputs)

    import numpy as np
    feature = np.random.uniform(size=(2,32,32,128))
    rois = np.concatenate([np.zeros((2,num_rois,2)), np.ones((2,num_rois,2))], axis=-1)   # [0,0,1,1]
    outputs = model.predict([feature, rois])
    print(outputs.shape)



