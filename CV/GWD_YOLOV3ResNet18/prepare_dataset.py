
# 多种操作系统接口
import os
import shutil
import random

# 提供了和 MATLAB 类似的绘图 API
import matplotlib.pyplot as plt
# python图像处理库中，Image是其中的核心库
from PIL import Image

import pandas as pd
# xml文件读写，文档对象模型接口的最小实现，parse可以解析一个XML文件通过文件或者文件名
from xml.dom.minidom import parse
# xml文件读写，文档对象模型接口的最小实现
import xml.dom.minidom

#科学计算
import numpy as np

# mindspore库
import mindspore as ms

# 将用户自定义的数据转为MindRecord格式数据集的类
from mindspore.mindrecord import FileWriter


# 筛选有效的图像文件，这些文件都在image_dir中。
# 输入参数为 image_dir，表示图像所在的文件夹路径
def filter_valid_data(image_dir, annotations_file=None, snum=0):
    # 定义一个字典 label_id，用于将目标类别转换为数字编码
    label_id={'background':0, 'wheat_head':1}

    # 定义一个空字典 image_dict，用于保存每个图像文件对应的标注信息
    # 定义一个空列表 image_files，用于保存所有有效的图像文件名
    image_dict = {}
    image_files=[]

    if not annotations_file:
        all_files = os.listdir(image_dir)
        for i in all_files:
            if (i[-3:]=='jpg' or i[-4:]=='jpeg') and i not in image_dict:
                image_files.append(i)
                label=[[0,0,0,0,0]]
                image_dict[i]=label
    else:
        df = pd.read_csv(annotations_file, encoding="utf-8")
        bboxs = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            df[column] = bboxs[:,i]
        df.drop(columns=['bbox'], inplace=True)

        all_files = os.listdir(image_dir)
        unique_files = df['image_id'].unique()
        print('unique_files: ', len(unique_files))

        if snum > 0:
            idx = list(range(len(unique_files)))
            sidx = random.sample(idx, snum)
            unique_files = unique_files[sidx]

        for img_nm in unique_files:
            file_name = img_nm + '.jpg'
            if not file_name in all_files:
                continue
            idx = df['image_id'] == img_nm
            bbox_dt = df.loc[idx]

            # 判断文件扩展名是否为 'jpg' 或 'jpeg'，并且文件名不在 image_dict 中
            if (file_name[-3:]=='jpg' or file_name[-4:]=='jpeg') and file_name not in image_dict:
                # 如果满足条件，则将文件名添加到 image_files 列表中
                image_files.append(file_name)
                # 定义一个空列表 label，用于保存当前文件的标注信息
                label=[]

                # 并将 label 保存到 image_dict 中，并跳过当前文件的处理
                if bbox_dt.shape[0] < 1:
                    label=[[0,0,0,0,0]]
                    image_dict[file_name]=label
                else:
                    for ind in bbox_dt.index:
                        temp=[]
                        # 获取目标类别，并将其转换为数字编码
                        name = 'wheat_head'
                        class_num = label_id[name]
                        # 获取目标框的坐标信息，并将其保存到 temp 中
                        xmin = bbox_dt['x'][ind]
                        ymin = bbox_dt['y'][ind]
                        xmax = bbox_dt['x'][ind] + bbox_dt['w'][ind]
                        ymax = bbox_dt['y'][ind] + bbox_dt['h'][ind]
                        temp.append(int(xmin))
                        temp.append(int(ymin))
                        temp.append(int(xmax))
                        temp.append(int(ymax))
                        temp.append(class_num)
                        # 将 temp 添加到 label 中
                        label.append(temp)
                    # 将 label 保存到 image_dict 中
                    image_dict[file_name]=label
        '''            
        print(image_files[0:5])
        for f in image_files[0:5]:
            print(image_dict[f])
        '''
    # 返回 image_files 和 image_dict
    return image_files, image_dict


# 通过image_dir创建MindRecord文件。
def data_to_mindrecord_byte_image(image_dir, mindrecord_dir, prefix, file_num, anno_file=None, snum=0):
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    image_files, image_anno_dict = filter_valid_data(image_dir, anno_file, snum)

    # 定义了一个yolo_json字典，描述了MindRecord的schema
    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
        "file": {"type": "string"},
    }
    # 添加schema到MindRecord中
    writer.add_schema(yolo_json, "yolo_json")

    # 对于每一个图像文件，把图像和标注信息打包成一个字典
    # 然后调用write_raw_data方法写入MindRecord
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name],dtype=np.int32)
        #print(annos.shape)
        row = {"image": img, "annotation": annos, "file": image_name}
        writer.write_raw_data([row])
    writer.commit()


def showImg(file_path, bboxs):
    fig = plt.figure()  # 相当于创建画板
    ax = fig.add_subplot(1, 1, 1)  # 创建子图，相当于在画板中添加一个画纸，当然可创建多个画纸，具体由其中参数而定
    f = Image.open(file_path)
    img_np = np.asarray(f, dtype=np.float32)    # H，W，C格式
    ax.imshow(img_np.astype(np.uint8))          # 当前画纸中画一个图片

    for box in bboxs:
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
    # 添加方框，(xmin,ymin)表示左顶点坐标，(xmax-xmin),(ymax-ymin)表示方框长宽
        ax.add_patch(
        plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin),
                      fill=False, edgecolor='red', linewidth=2))
    plt.show()

# ---------------------------------------------------------------------------------------------------------
# Global_wheat_detection dataset from: https://www.kaggle.com/competitions/global-wheat-detection/data
# ---------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    dir = "/media/hhj/localssd/DL_data/Crop_and_weeds/Global_wheat_detection"
    mindrecord_dir_train = './mindrecord/train'
    prefix = "yolo.mindrecord"
    train_annotations = os.path.join(dir, "train.csv")
    image_dir= os.path.join(dir, "train")

    if os.path.exists(mindrecord_dir_train):
        shutil.rmtree(mindrecord_dir_train)

    if os.path.exists(mindrecord_dir_train) and os.listdir(mindrecord_dir_train):
        print('The mindrecord file had exists!')
    else:
        image_dir = os.path.join(dir, "train")

    if not os.path.exists(mindrecord_dir_train):
        os.makedirs(mindrecord_dir_train)

    '''
    img_files, img_dict = filter_valid_data(image_dir, train_annotations)
    for i in range(10):
        image_path = os.path.join(dir, "train",  img_files[i])
        bboxs = img_dict[img_files[i]]
        showImg(image_path, bboxs)
    '''
    print("Create Mindrecord.")
    data_to_mindrecord_byte_image(image_dir, mindrecord_dir_train, prefix, 1, train_annotations, 640)
    print("Create Mindrecord Done, at {}".format(mindrecord_dir_train))

    exit(0)
