reference: https://zhuanlan.zhihu.com/p/31426458
repo: https://github.com/pytorch/vision/blob/43d772067fe77965ec8fc49c799de5cea44b8aa2/torchvision/models/detection/faster_rcnn.py

## quickly look back
    RCNN: 
        1. 原图通过selective search找到proposals，
        2. 在原图上裁剪ROI，resize，然后经过CNN提取特征，
        3. 分别进行SVM分类和bbox回归

    Fast RCNN: 
        1. 原图通过selective search找到proposals，
        2. 原图经过CNN得到特征图，
        3. 在特征图上裁剪ROI，ROIPooling到固定特征图尺寸，
        4. 全连接层得到特征向量，
        5. 分别进行softmax分类和bbox回归
        * 核心创新点在CNN提取特征只需做一次，而不是每个proposal做一次

    Faster RCNN: 
        1. 原图经过CNN得到特征图，
        2. 通过RPN head找到proposals，
        3. 在特征图上裁剪ROI，ROIPooling到固定特征图尺寸，
        4. 分别进行softmax分类和bbox回归
        * 核心创新点在RPN

    Mask-RCNN:
        1. 原图经过CNN得到特征图，这里backbone用了resnet-fpn，因为还有分割任务
        2. 通过RPN head找到proposals，
        3. 在特征图上裁剪ROI，ROIAlign到固定特征图尺寸，
        4. detector branch，还是跟FasterRCNN一样
        5. seg branch，全卷积，特征图尺度的heatmap
        * 核心创新点在ROIAlign和mask branch


## Faster RCNN

### architecture
    两个模块：
    * RPN
    * Fast R-CNN detector

    4-step交替训练：
    1. 训练RPN
    2. 用上一步RPN的结果训练独立的detector (冻住shared back+rpn)
    3. 用上一步的detector初始化RPN，冻住shared back+detector，只fine-tune RPN
    4. 然后再冻住RPN，只fine-tune detector

    没有BN
    * 2016年的论文，没到那个年代
    * batch size较小（1/2），采样postive和计算topK都是per sample的，没share among data，或许可以考虑IN？


### shared vgg back
    basic block: conv-relu & maxpooling
    13 conv layers: [2, 2, 3, 3, 3]
    stride 16: [2,2,2,2,1]，去掉了block5的maxpooling
    pretrained ImageNet weights


### RPN
    input: raw image of any size
    outputs: a set of rectangular object proposals with objectness scores

    nearly cost-free RPN: 全卷积
    * 3x3 conv + relu, channel 256/512
    * 然后接两个分支conv：
        box branch：1x1 conv，channel 4k
        cls branch：1x1 conv，channel 2k with softmax / channel k with sigmoid
        每个location最多预测k个proposals
    * zero-mean & 0.01-std Gaussian初始化

    postprocess on rpn网络的raw outputs
    * take topk boxes
    * clip boundary
    * remove small boxes
    * remove low score boxes
    * nms
    * keep topk boxes

    anchors
    * 每个location，有k个anchor，
    * anchor以loc中心为中心，of multiple scales and ratios，enable to predict multi-scale objects
    * k=9
    * 所有cross-boundary的anchors忽略不算，不然会引入较多且较难回归的框

    binary classification for each anchor
    1. 和每个gt box的IoU最大的anchor为positive
    2. 和任意gt box的IoU大于0.7的anchor为postive
    3. 其余的anchor，如果和任意gt box的IoU小于0.3，为negative
    4. 再其余的anchor为ignore，在训练分类的时候不回传梯度
    * log loss
    * normed by mini-batch size (256)

    regression
    1. 只针对正样本计算regression loss，因为只有正样本有gt value
    2. 优化目标是：pred box相对于anchor box的相对量，去fit gt box相对于anchor box的相对量
    * smooth L1
    * normed by locations (2400)
    * reweighting factor=10, torch vision里面是[1,1,1,1], 但是我实验下来一阶段加权比不加权出框多

    regression targets:
    txty: (gt_center_xy - anchor_center_xy) / anchor_wh
    twth: log(gt_wh/anchor_wh)
    * 有正有负且无界，所以box branch输出层没有激活函数
    * 线性回归：因为positive anchor与gt box比较接近，可以认为是线性关系

    loss
    * randomly sample 256 anchors with balanced positives & negatives per image
    * 如果正样本少于128个，就用负样本填充
    * SGD：momentum=0.9，weight decay=5e-4
    * lr=1e-3 for 60k，lr=1e-4 for next 20k

    todolist:
    1. 将postprocess加进去
    2. 将gt targets的计算加进postprocess层里面去
    3. 将loss封装成layer加进去


### RPNProposal
    inputs: 
        rpn outputs, [b,hs,ws,a,1] rpn_objectness & [b,hs,ws,a,4] rpn_boxoffset 
        gt boxes, [b,M,c+1+4]
    outputs: proposal boxes, [b,N,4], x1y1x2y2

    训练过程
    loop image: 
    1. 计算iou，为每个proposal找到最match的gt box，及其label
    2. 低于给定阈值的matched iou，proposal的label设定为0/-1
    3. 在proposals中采样，选取固定比例的positive & negative
    4. 剩下的positive & negative样本进行encode，regression targets是proposal和gt的偏移量


### ROIPooling
    given featuremap: [b,h,w,c] & rois: [b,N,4]
    对每个roi，找到其在featuremap level上的整数坐标（一次近似），然后切分成整数坐标的bins（两次近似），
    在每个bin范围内做maxpooling
    在代码实现的时候需要一个框一个框遍历，有没有更高效的实现方法？？


### Detector
    adopt Fast-RCNN
    a view from Fast RCNN：detector训练阶段batch size不要开太大
    - 因为featuremap的计算量随着batch size增大而增大，
    - 而我们提取的topK平均给每个样本则越来越少，相当于浪费CNN计算一次只用一点点信息，不划算
    - batch=2, RoIs=128/2 per image
    - SGD

    ROI pooling, 得到统一尺寸的特征图, 7x7: [b,N,7,7,1024]
    shared fcs: fc-relu-fc-reulu, dim=2048, [b,N,2048]
    individual fc branch:
        box branch: fc, dim=N*4, per-cls box offsets pred
        cls branch: fc+softmax, dim=N+1, cls among N+1
    可以看到主要参数量在Detector的fc中

    multi-classification for each proposals
    1. 计算proposals和gt boxes之间的iou
    2. box_bg_iou_thresh: 0.5，与gt box的iou大于0.5为positive
    3. box_fg_iou_thresh: 0.5，与gt box的iou小于0.5为negative
    4. positive_fraction: 0.25，proposals的正负样本比例，跟rpn阶段类似
    5. proposals per image: 512，按照上述比例random select

    * reweighting factor=10, torch vision里面是[10,10,5,5]，位移是10长宽是5

    第二步训练Detector head的时候一定要冻住backbone和rpn，提供稳定、固定的proposals，不然分分钟训飞了



### experiment details
    pre-processing: 
        mean & std: a fixed mean tuple (mean/std values of the dataset)
        norm: [0,1]
        rescale: making the shorter side=600，max side<=1000
        aug: 水平翻转

    初始化: 
        all new layers' weights: zero-mean & 0.01-std Gaussian

    train & test time NMS on RPN: 
    * IoU thresh=0.7
    * pre_nms_top_n=2000
    * post_nms_top_n=2000
    实际实验发现，在rpn proposals里面做了NMS以后，前景变得特别少

    test time postprocess on detector head:
    1. 预测所有的proposals
    2. remove low scoring boxes: score_thresh=0.05
    3. remove empty boxes: minsize=1e-2
    4. nms per class: nms_thresh=0.5, max_detections per image=100

    set use_multiprocessing=False under Win

    pkg version
    * tf 1.13.1
    * keras 2.2.4
    * CUDA 10.0
    * cuDNN 7.4
    * numpy 1.16


### further
    torchvision里面除了basic版本，还有使用mobilenet_v3 back和fpn的版本




















