from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, \
                         add, ReLU, BatchNormalization, Lambda
from keras.models import Model
from keras.initializers import RandomNormal
import keras.backend as K
from utils.backbone import vgg_back, resnet_back
from utils.ROIPooling import RoiPooling
from utils.ROIAlign import RoiAlign
from utils.RPNTargets import RPNTargets
from utils.RPNProposal import RPNProposal
from loss import rpnet_loss, detection_loss
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
KERNEL_INITIALIZER = RandomNormal(mean=0.0, stddev=0.01)


def fasterRCNN(input_shape=(None,None,3), n_classes=80, n_anchors=9, mode='train'):

    inpt = Input(input_shape)

    # shared back: s16 feature map, [b,h,w,c]
    shared_feature = resnet_back(input_shape)(inpt)
    filters_in = K.int_shape(shared_feature)[-1]

    # rpn: feature grid level prediction, [b,h,w,k,1] & [b,h,w,k,4]
    rpn_objectness, rpn_boxoffset = rpn(filters_in, n_anchors=n_anchors)(shared_feature)

    # roi: [b,N,4] & [b,N,c+1+4c]
    if mode=='train':
        # gt input
        gt_inpt = Input((None, n_classes+1+4))   # gt boxes, [b,M,c+1+4]
        # rpn_loss
        rpn_targets = RPNTargets(input_hw=input_shape[:2])(gt_inpt)   # [b,h,w,a,1+4], anchor offsets
        rpn_loss = Lambda(rpnet_loss)([rpn_objectness, rpn_boxoffset, rpn_targets])
        # rpn_model
        rpn_model = Model([inpt, gt_inpt], rpn_loss)
        # detection head: kernel grid level prediction, [b,N,c+1] & [b,N,4c]
        rois, roi_targets = RPNProposal(n_classes=n_classes, n_anchors=n_anchors, mode=mode,
                                        top_n=500, positive_fraction=0.5, batch_size_per_img=200)(
                                        [rpn_objectness, rpn_boxoffset, gt_inpt])
        cls_output, box_output = detector(filters_in, n_classes=n_classes)([shared_feature, rois])
        # detection loss
        detector_loss = Lambda(detection_loss, arguments={'n_classes': n_classes})([cls_output, box_output, roi_targets])
        # detection model
        detection_model = Model([inpt, gt_inpt], detector_loss)

    else:   # 'test' mode
        # proposals
        rois = RPNProposal(n_classes=n_classes, n_anchors=n_anchors, mode=mode, top_n=200)(
                          [rpn_objectness, rpn_boxoffset])
        # rpn_model
        rpn_model = Model(inpt, [rpn_objectness, rpn_boxoffset, rois])
        # detection model
        cls_output, box_output = detector(filters_in, n_classes=n_classes)([shared_feature, rois])
        detection_model = Model(inpt, [cls_output, box_output, rois])

    return rpn_model, detection_model


def rpn(filters_in=512, n_anchors=9):
    inpt = Input((None,None,filters_in))

    x = Conv2D(512, 3, strides=1, padding='same', activation='relu', kernel_initializer=KERNEL_INITIALIZER)(inpt)
    cls_output = Conv2D(n_anchors*1, 1, activation='sigmoid', kernel_initializer=KERNEL_INITIALIZER)(x)
    box_output = Conv2D(n_anchors*4, 1, activation=None, kernel_initializer=KERNEL_INITIALIZER)(x)

    # reshape
    cls_output = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], n_anchors, 1)))(cls_output)
    box_output = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], n_anchors, 4)))(box_output)

    model = Model(inpt, [cls_output, box_output], name='rpn')
    return model


def detector(filters_in=512, n_classes=80):
    shared_feature = Input((None,None,filters_in))
    rois = Input((None,4))

    pooled_feature = RoiAlign(pool_size=7,)([shared_feature, rois])   # [b,topK,7,7,c]
    x = TimeDistributed(Flatten())(pooled_feature)   # [b,N,flatten]
    x = TimeDistributed(Dense(4096, activation='relu', kernel_initializer=KERNEL_INITIALIZER))(x)
    x = TimeDistributed(Dense(4096, activation='relu', kernel_initializer=KERNEL_INITIALIZER))(x)
    cls_output = TimeDistributed(Dense(n_classes+1, activation='softmax', kernel_initializer=KERNEL_INITIALIZER))(x)
    box_output = TimeDistributed(Dense(4*n_classes, activation='linear', kernel_initializer=KERNEL_INITIALIZER))(x)

    model = Model([shared_feature, rois], [cls_output, box_output], name='detector')
    return model


if __name__ == '__main__':

    rpn_model, detection_model = fasterRCNN((512,512,3), mode='train')

    rpn_model.summary()

    print("=========== rpn_model ===========")
    for layer in rpn_model.layers:
        if 'vgg' in layer.name:
            layer.trainable = False
        print(layer.name, layer.trainable)

    print("=========== entire_model ===========")
    for layer in detection_model.layers:
        print(layer.name, layer.trainable)














