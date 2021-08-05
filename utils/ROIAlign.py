from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class RoiAlign(Layer):
    '''
    # inputs
        feature map: [b,h,w,c]
        rpn pred: [b,N,4], normed, x1y1x2y2
    # outputs
        proposal featuremaps: pooling maps, [b,topK,7,7,c]
    '''

    def __init__(self, pool_size=7, **kwargs):
        super(RoiAlign, self).__init__(**kwargs)
        self.pool_size = pool_size

    def build(self, input_shape):
        self.n_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        n_rois = input_shape[1][1]
        return (None, n_rois, self.pool_size, self.pool_size, self.n_channels)

    def call(self, x, mask=None):

        assert(len(x) == 2)

        feature_map = x[0]
        rpn_pred = x[1]       # [b,N,4]

        # loop b
        def loop_body(b, roi_feature):
            crops = self.crop_and_resize(feature_map[b], rpn_pred[b])   # [N,7,7,c]
            roi_feature = roi_feature.write(b, crops)
            return b+1, roi_feature
        batch = K.shape(feature_map)[0]     # batch size
        roi_feature = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
        _, roi_feature = tf.while_loop(lambda b,*args: b<batch, loop_body, [0, roi_feature])
        roi_feature = roi_feature.stack()    # [b,N,7,7,c]

        return roi_feature

    def crop_and_resize(self, feature_map, rpn_pred):
        # for a single image
        # featuremap: [h,w,c]
        # boxes: [N,4]
        # returns: proposal maps, [N,7,7,c]

        feature_map = K.expand_dims(feature_map, axis=0)
        box_y1x1y2x2 =  K.concatenate([rpn_pred[...,1::-1],rpn_pred[...,3:1:-1]], axis=-1)   # [N,4], normed
        box_indices = tf.zeros((K.shape(rpn_pred)[0],), dtype='int32')
        final_output = tf.image.crop_and_resize(feature_map, box_y1x1y2x2, box_indices, (self.pool_size, self.pool_size), method="bilinear")

        return final_output


if __name__ == '__main__':

    from keras.layers import Input
    from keras.models import Model

    feature = Input((512,512,3))
    rois = Input((None,4))
    outputs = RoiAlign(pool_size=7)([feature, rois])
    print(outputs)
    model = Model([feature, rois], outputs)

    # test roialign
    import numpy as np
    import json
    import cv2

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

    boxes = get_box("../data/tux_positive.json")
    boxes = np.expand_dims(boxes[:,:4],0)
    boxes = np.concatenate([boxes, np.array([[[0,0,1,1]]])], axis=1)
    img = cv2.imread("../data/tux_positive.jpg", 1)
    img = cv2.resize(img, (512,512))
    img = img.reshape((1,512,512,3))

    preds = model.predict([img, boxes])[0]
    for poolingmap in preds:
        print(poolingmap.shape)   # 'f32'
        tmp = cv2.resize(poolingmap, (256,256))
        cv2.imshow("tmp", tmp/255.)
        cv2.waitKey(0)







