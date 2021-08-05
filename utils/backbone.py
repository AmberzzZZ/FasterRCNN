from keras.layers import Input, Conv2D, MaxPooling2D, add, ReLU, BatchNormalization
from keras.models import Model


def vgg_back(input_shape=(None,None,3)):

    inpt = Input(input_shape)

    # block1
    x = Conv2D(64, 3, activation='relu', padding='same')(inpt)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # block2
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # block3
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # block4
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    # block5
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    shared_feature = Conv2D(512, (3, 3), activation='relu', padding='same')(x)   # [b,h,w,c]

    model = Model(inpt, shared_feature, name='vgg')
    # model.load_weights("weights/vgg.h5", by_name=True, skip_mismatch=True)

    return model


def resnet_back(input_shape=(None,None,3), depth=50):

    n_blocks = {50: [3,4,6,3], 101: [3,4,23,3]}
    n_filters = [256, 512, 1024, 2048]

    inpt = Input(input_shape)

    # stem: conv+bn+relu+pool
    x = Conv_BN(inpt, 64, 7, strides=2, activation='leaky')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # blocks
    num_blocks = n_blocks[depth][:3]  # 1-3 stages, output stride=16
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            strides = 2 if i and j==0 else 1  # downsamp: 1st block for every stage
            x = res_block(x, n_filters[i], strides)

    # model
    model = Model(inpt, x, name='resnet')
    # model.load_weights("weights/r50.h5", by_name=True, skip_mismatch=True)

    return model


def res_block(x, n_filters, strides):
    inpt = x
    # residual
    x = Conv_BN(x, n_filters//4, 1, strides=strides, activation='relu')
    x = Conv_BN(x, n_filters//4, 3, strides=1, activation='relu')
    x = Conv_BN(x, n_filters, 1, strides=1, activation=None)
    # shortcut
    if strides!=1 or inpt._keras_shape[-1]!=n_filters:
        inpt = Conv_BN(inpt, n_filters, 1, strides=strides, activation=None)
    x = add([inpt, x])
    x = ReLU()(x)
    return x


def Conv_BN(x, n_filters, kernel_size, strides, activation=None):
    x = Conv2D(n_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    return x


if __name__ == '__main__':

    pass


