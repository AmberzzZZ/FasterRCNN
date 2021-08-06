from fasterRCNN import fasterRCNN
from dataSequence import dataSequence, category_name
from config import config
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.utils import multi_gpu_model
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_cnt = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


if __name__ == '__main__':

    img_dir = "data/"
    label_dir = "data/"
    weight_dir = "weights"

    input_shape = (512,512,3)
    n_classes = 2
    output_stride = 16
    n_anchors = config.n_anchors

    # model
    rpn_model, detection_model = fasterRCNN(input_shape, n_classes, n_anchors, mode='train')

    # train
    step = 2
    if step==1:
        # step1: pretrain & std init, train rpn_model
        batch_size = 32
        train_generator = dataSequence(img_dir, label_dir, n_classes, output_stride=output_stride,
                                       input_shape=input_shape, batch_size=batch_size)
        rpn_model.compile(Adam(1e-3), loss=lambda y_true,y_pred: y_pred)
        filepath = weight_dir + "/rpn_epoch_{epoch:02d}_loss_{loss:.3f}.h5"
        if gpu_cnt>1:
            checkpoint = ParallelModelCheckpoint(rpn_model, filepath, monitor='loss', verbose=1, mode='auto')
            rpn_model = multi_gpu_model(rpn_model, gpu_cnt)
            rpn_model.compile(Adam(1e-3), loss=lambda y_true,y_pred: y_pred)
        else:
            checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto')
        rpn_model.fit_generator(train_generator,
                                steps_per_epoch=1,
                                epochs=1,
                                initial_epoch=0,
                                workers=16,
                                use_multiprocessing=False,
                                callbacks=[checkpoint],
                                verbose=1)

    elif step==2:
        # step2: pretrain & std init, use step1 proposals, train detection_model
        batch_size = 8
        train_generator = dataSequence(img_dir, label_dir, n_classes, output_stride=output_stride,
                                       input_shape=input_shape, batch_size=batch_size)
        detection_model.load_weights("rpn.h5", by_name=True, skip_mismatch=True)
        # freeze rpn heads
        for layer in detection_model.layers:
            if layer.name in ['rpn', 'rpn_back']:
                layer.trainable = False
        detection_model.compile(Adam(3e-4), loss=lambda y_true,y_pred: y_pred)
        filepath = weight_dir + "/detector_epoch_{epoch:02d}_loss_{loss:.3f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto')
        detection_model.fit_generator(train_generator,
                                      steps_per_epoch=1,
                                      epochs=1,
                                      initial_epoch=0,
                                      workers=16,
                                      use_multiprocessing=False,
                                      callbacks=[checkpoint],
                                      verbose=1)
    elif step==3:
        # step3: load step2 weights, freeze backbone, train rpn head
        batch_size = 32
        rpn_model.load_weights("detector.h5", by_name=True, skip_mismatch=True)
        for layer in rpn_model.layers:
            if layer.name in ['rpn_back']:
                layer.trainable = False
        rpn_model.compile(Adam(1e-4), loss=lambda y_true,y_pred: y_pred)
        filepath = weight_dir + "/rpn_ft_epoch_{epoch:02d}_loss_{loss:.3f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto')
        rpn_model.fit_generator(train_generator,
                                steps_per_epoch=1,
                                epochs=1,
                                initial_epoch=0,
                                workers=16,
                                use_multiprocessing=False,
                                callbacks=[checkpoint],
                                verbose=1)
    elif step==4:
        # step4: load step3 weights, freeze backbone, use step3 proposals, ttrain detection head
        batch_size = 8
        detection_model.load_weights("rpn_ft.h5", by_name=True, skip_mismatch=True)
        for layer in detection_model.layers:
            if layer.name in ['rpn', 'rpn_back', 'det_back']:
                layer.trainable = False
        detection_model.compile(Adam(1e-4), loss=lambda y_true,y_pred: y_pred)
        filepath = weight_dir + "/detector_ft_epoch_{epoch:02d}_loss_{loss:.3f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='auto')
        detection_model.fit_generator(train_generator,
                                      steps_per_epoch=1,
                                      epochs=1,
                                      initial_epoch=0,
                                      workers=16,
                                      use_multiprocessing=False,
                                      callbacks=[checkpoint],
                                      verbose=1)



