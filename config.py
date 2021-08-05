import numpy as np


def generate_anchors(anchor_scales, anchor_ratios):
    anchors = np.array(anchor_scales).reshape((-1,1)).astype(np.float32)
    anchors = np.tile(anchors, (len(anchor_ratios),2))

    factor = np.tile(np.array(anchor_ratios).reshape((-1,1)), len(anchor_scales)).reshape((-1, 1))

    anchors /= np.sqrt(factor)
    anchors[...,:1] *= factor

    return anchors    # [N,2], wh


class config():

    # anchor_scales = [256,416,448]   # 3
    anchor_scales = [128, 256, 512]
    anchor_ratios = [0.5, 1., 2]   # 3
    anchors = generate_anchors(anchor_scales, anchor_ratios)         # [9,2]
    n_anchors = anchors.shape[0]


if __name__ == '__main__':

    anchors = generate_anchors(config.anchor_scales, config.anchor_ratios)
    print(anchors)


