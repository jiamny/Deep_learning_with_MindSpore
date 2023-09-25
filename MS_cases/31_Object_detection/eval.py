
from train import ConfigYOLOV3ResNet18, create_yolo_dataset,  YoloWithEval, yolov3_resnet18, data_to_mindrecord_byte_image
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


## 模型预测
# Test for yolov3-resnet18
# 系统模块
import os
# 时间
import time
# 提供了和 MATLAB 类似的绘图 API
import matplotlib.pyplot as plt
# python图像处理库中，Image是其中的核心库
from PIL import Image
# 科学计算库
import numpy as np
# 提供了许多函数和变量来处理 Python 运行时环境的不同部分
import sys

# 你的代码路径
sys.path.insert(0, './code/')

from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# 应用NMS至bbox
def apply_nms(all_boxes, all_scores, thres, max_boxes):
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    #print('x1: ', x1, ' y1: ', y1, ' x2: ', x2, ' y2: ', y2)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        #print('w: ', w, ' h: ', h)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


# 计算预测bbox的precision和recall
def tobox(boxes, box_scores):
    config = ConfigYOLOV3ResNet18()
    num_classes = config.num_classes
    mask = box_scores >= config.obj_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    max_boxes = config.nms_max_num
    for c in range(num_classes):
        class_boxes = np.reshape(boxes, [-1, 4])[np.reshape(mask[:, c], [-1])]
        class_box_scores = np.reshape(box_scores[:, c], [-1])[np.reshape(mask[:, c], [-1])]
        nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, max_boxes)
        # nms_index = apply_nms(class_boxes, class_box_scores, 0.5, max_boxes)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes = np.concatenate(boxes_, axis=0)
    classes = np.concatenate(classes_, axis=0)
    scores = np.concatenate(scores_, axis=0)

    return boxes, classes, scores


# Yolov3评估
def yolo_eval(cfg_test):
    ds = create_yolo_dataset(cfg_test.mindrecord_file, batch_size=1, is_training=False)
    config = ConfigYOLOV3ResNet18()
    net = yolov3_resnet18(config)
    eval_net = YoloWithEval(net, config)
    print("Load Checkpoint!")
    print(cfg_test.ckpt_path)
    param_dict = load_checkpoint(cfg_test.ckpt_path)
    load_param_into_net(net, param_dict)

    eval_net.set_train(False)
    i = 1.
    total = ds.get_dataset_size()
    start = time.time()
    pred_data = []
    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")

    num_class = {0: 'person', 1: 'face', 2: 'mask'}
    for data in ds.create_dict_iterator(output_numpy=True):
        # print("------------------------开始------------------")
        img_np = data['image']
        image_shape = data['image_shape']
        # print("image_shape", image_shape)
        annotation = data['annotation']
        image_file = data['file']
        image_file = image_file.tobytes().decode('ascii')

        eval_net.set_train(False)
        output = eval_net(Tensor(img_np), Tensor(image_shape))
        for batch_idx in range(img_np.shape[0]):
            boxes = output[0].asnumpy()[batch_idx]
            box_scores = output[1].asnumpy()[batch_idx]
            image = img_np[batch_idx, ...]
            boxes, classes, scores = tobox(boxes, box_scores)
            # print(classes)
            # print(scores)
            fig = plt.figure()  # 相当于创建画板
            ax = fig.add_subplot(1, 1, 1)  # 创建子图，相当于在画板中添加一个画纸，当然可创建多个画纸，具体由其中参数而定

            # image_path = os.path.join(cfg.image_dir, image_file)
            # print(cfg.image_dir)
            # print(image_file)
            image_path = cfg_test.image_dir+'/'+image_file
            # print(image_path)
            new_path = ''
            for i in image_path:
                # print(i,ord(i))
                # 存在大量的ord值为0的
                if ord(i)!=0:
                    new_path+=i
            # print(new_path)
            f = Image.open(new_path)
            # f = Image.open(image_path)
            img_np = np.asarray(f, dtype=np.float32)    # H，W，C格式
            ax.imshow(img_np.astype(np.uint8))          # 当前画纸中画一个图片

            for box_index in range(boxes.shape[0]):
                ymin = boxes[box_index][0]
                xmin = boxes[box_index][1]
                ymax = boxes[box_index][2]
                xmax = boxes[box_index][3]

                # 添加方框，(xmin,ymin)表示左顶点坐标，(xmax-xmin),(ymax-ymin)表示方框长宽
                ax.add_patch(
                    plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                                  fill=False, edgecolor='red', linewidth=2))
                # 给方框加标注，xmin,ymin表示x,y坐标，其它相当于画笔属性

                ax.text(xmin, ymin - 2, s=str(num_class[classes[box_index]]) + ": {score:.2f}".format(score=scores[box_index]),
                        style='italic', bbox={'facecolor': 'blue', 'alpha': 0.9, 'pad': 0},
                        fontdict=dict(fontsize=10, color='w',
                                      family='monospace',   #字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                                      weight='normal'       #磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'

                                      ))
            plt.show()
        # print("------------------------结束------------------")
# ---------------yolov3  test-------------------------

class cfg_test:
    device_id = 0
    ckpt_url = 'output'
    train_url = './testoutput'
    image_dir = './data/test'

if __name__ == '__main__':
    cfg_test = cfg_test()

    ckpt_path = './ckpt/'
    cfg_test.ckpt_path = os.path.join(ckpt_path, "yolov3-270_80.ckpt") #此处文件名需要随实际训练epoch情况修改
    data_path = './data/'

    mindrecord_dir_test = os.path.join(data_path,'mindrecord/test')
    prefix = "yolo.mindrecord"
    cfg_test.mindrecord_file = os.path.join(mindrecord_dir_test, prefix)
    cfg_test.image_dir = os.path.join(data_path, "test")
    if os.path.exists(mindrecord_dir_test) and os.listdir(mindrecord_dir_test):
        print('The mindrecord file had exists!')
    else:
        if not os.path.isdir(mindrecord_dir_test):
            os.makedirs(mindrecord_dir_test)
        prefix = "yolo.mindrecord"
        cfg_test.mindrecord_file = os.path.join(mindrecord_dir_test, prefix)
        print("Create Mindrecord.")
        data_to_mindrecord_byte_image(cfg_test.image_dir, mindrecord_dir_test, prefix, 1)
        print("Create Mindrecord Done, at {}".format(mindrecord_dir_test))
    print("Start Eval!")

    yolo_eval(cfg_test)
    exit(0)