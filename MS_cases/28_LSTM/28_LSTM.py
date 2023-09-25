
#  基于MindSpore实现LSTM算法 

## 模型构建

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train import Model
from mindspore import context


#设置参数变量
from easydict import EasyDict as edict
lstm_cfg = edict({
    'num_classes': 2,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'num_epochs': 1,    
    'batch_size': 128,
    'embed_size': 300,
    'num_hiddens': 100,
    'num_layers': 2,
    'bidirectional': True,
    'save_checkpoint_steps': 390,
    'keep_checkpoint_max': 10
})
args_train = edict({
    'preprocess': 'true',
    'aclimdb_path': "/media/hhj/localssd/DL_data/IMDB/aclImdb",
    'glove_path': "/media/hhj/localssd/DL_data/IMDB/glove",
    'preprocess_path': "./preprocess",
    'ckpt_path': "./",
    'pre_trained': None,
    'device_target': "CPU",
})
args_test = edict({
    'preprocess': 'false',
    'aclimdb_path': "/media/hhj/localssd/DL_data/IMDB/aclImdb",
    'glove_path': "/media/hhj/localssd/DL_data/IMDB/glove",
    'preprocess_path': "./preprocess",
    'ckpt_path': "./lstm-1_195.ckpt",
    'pre_trained': None,
    'device_target': "CPU",
})


import os
from itertools import chain
import numpy as np
import gensim
class ImdbParser():
    def __init__(self, imdb_path, glove_path, embed_size=300):
        self.__segs = ['train', 'test']
        self.__label_dic = {'pos': 1, 'neg': 0}
        self.__imdb_path = imdb_path
        self.__glove_dim = embed_size
        self.__glove_file = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')
        self.__imdb_datas = {}
        self.__features = {}
        self.__labels = {}
        self.__vacab = {}
        self.__word2idx = {}
        self.__weight_np = {}
        self.__wvmodel = None
#解析IMDB数据集，生成特征、标签和权重矩阵
    def parse(self):
        self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(self.__glove_file)
        for seg in self.__segs:
# 解析IMDB数据
            self.__parse_imdb_datas(seg)
# 解析特征和标签
            self.__parse_features_and_labels(seg)
# 生成权重数组
            self.__gen_weight_np(seg)
#解析IMDB数据文件，获取文本和标签对应的列表
    def __parse_imdb_datas(self, seg):
        data_lists = []
        for label_name, label_id in self.__label_dic.items():
            sentence_dir = os.path.join(self.__imdb_path, seg, label_name)
            for file in os.listdir(sentence_dir):
                with open(os.path.join(sentence_dir, file), mode='r', encoding='utf8') as f:
                    sentence = f.read().replace('\n', '')
                    data_lists.append([sentence, label_id])
        self.__imdb_datas[seg] = data_lists
    def __parse_features_and_labels(self, seg):
        features = []
        labels = []
        for sentence, label in self.__imdb_datas[seg]:
            features.append(sentence)
            labels.append(label)
        self.__features[seg] = features
        self.__labels[seg] = labels
# 更新特征为令牌化形式
        self.__updata_features_to_tokenized(seg)
# 解析词汇表
        self.__parse_vacab(seg)
# 编码特征
        self.__encode_features(seg)
# 填充特征
        self.__padding_features(seg)
#将特征（文本）转换为令牌化形式
    def __updata_features_to_tokenized(self, seg):
        tokenized_features = []
        for sentence in self.__features[seg]:
            tokenized_sentence = [word.lower() for word in sentence.split(" ")]
            tokenized_features.append(tokenized_sentence)
        self.__features[seg] = tokenized_features
#解析词汇表，生成词汇表和单词到索引的映射
    def __parse_vacab(self, seg):
        tokenized_features = self.__features[seg]
        vocab = set(chain(*tokenized_features))
        self.__vacab[seg] = vocab
        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx[seg] = word_to_idx
#将特征（令牌化形式）编码为索引序列
    def __encode_features(self, seg):
        word_to_idx = self.__word2idx['train']
        encoded_features = []
        for tokenized_sentence in self.__features[seg]:
            encoded_sentence = []
            for word in tokenized_sentence:
                encoded_sentence.append(word_to_idx.get(word, 0))
            encoded_features.append(encoded_sentence)
        self.__features[seg] = encoded_features
#填充特征序列，使其具有相同的长度
    def __padding_features(self, seg, maxlen=500, pad=0):
        padded_features = []
        for feature in self.__features[seg]:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while len(padded_feature) < maxlen:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        self.__features[seg] = padded_features
#生成权重矩阵，用于将词汇转换为预训练的词向量
    def __gen_weight_np(self, seg):
        weight_np = np.zeros((len(self.__word2idx[seg]), self.__glove_dim), dtype=np.float32)
        for word, idx in self.__word2idx[seg].items():
            if word not in self.__wvmodel:
                continue
            word_vector = self.__wvmodel.get_vector(word)
            weight_np[idx, :] = word_vector
        self.__weight_np[seg] = weight_np
#获取指定数据集的特征、标签和权重矩阵
    def get_datas(self, seg):
        features = np.array(self.__features[seg]).astype(np.int32)
        labels = np.array(self.__labels[seg]).astype(np.int32)
        weight = np.array(self.__weight_np[seg])
        return features, labels, weight
import os
import numpy as np
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
#创建MindSpore数据集
def lstm_create_dataset(data_home, batch_size, repeat_num=1, training=True):
    ds.config.set_seed(1)
    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord0")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord0")
# 从MindRecord文件中读取数据集
    data_set = ds.MindDataset(data_dir, columns_list=["feature", "label"], num_parallel_workers=4)
# 对数据集进行洗牌
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
# 按批次进行数据集划分
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
# 对数据集进行重复
    data_set = data_set.repeat(count=repeat_num)
    return data_set
# 将特征和标签转换为MindRecord文件格式
def _convert_to_mindrecord(data_home, features, labels, weight_np=None, training=True):
    if weight_np is not None:
        np.savetxt(os.path.join(data_home, 'weight.txt'), weight_np)
    schema_json = {"id": {"type": "int32"},
                   "label": {"type": "int32"},
                   "feature": {"type": "int32", "shape": [-1]}}
    data_dir = os.path.join(data_home, "aclImdb_train.mindrecord")
    if not training:
        data_dir = os.path.join(data_home, "aclImdb_test.mindrecord")
    def get_imdb_data(features, labels):
        data_list = []
        for i, (label, feature) in enumerate(zip(labels, features)):
            data_json = {"id": i,
                         "label": int(label),
                         "feature": feature.reshape(-1)}
            data_list.append(data_json)
        return data_list
    writer = FileWriter(data_dir, shard_num=4)
# 将数据转换为MindRecord的数据格式
    data = get_imdb_data(features, labels)
# 添加schema和索引
    writer.add_schema(schema_json, "nlp_schema")
    writer.add_index(["id", "label"])
    writer.write_raw_data(data)
    writer.commit()
# 将数据集转换为MindRecord文件格式
def convert_to_mindrecord(embed_size, aclimdb_path, preprocess_path, glove_path):
    parser = ImdbParser(aclimdb_path, glove_path, embed_size)
    parser.parse()
    if not os.path.exists(preprocess_path):
        print(f"preprocess path {preprocess_path} is not exist")
        os.makedirs(preprocess_path)
#获取训练集数据
    train_features, train_labels, train_weight_np = parser.get_datas('train')
    _convert_to_mindrecord(preprocess_path, train_features, train_labels, train_weight_np)
# 获取测试集数据
    test_features, test_labels, _ = parser.get_datas('test')
    _convert_to_mindrecord(preprocess_path, test_features, test_labels, training=False)


import mindspore.nn as nn
import mindspore.ops as ops
class SentimentNet(nn.Cell):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 weight,
                 batch_size):
        super(SentimentNet, self).__init__()
# 创建嵌入层
        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
# 转置操作
        self.trans = ops.Transpose()
        self.perm = (1, 0, 2)
# 创建LSTM编码器
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               has_bias=True,
                               bidirectional=bidirectional,
                               dropout=0.0)
# 拼接操作
        self.concat = ops.Concat(1)
# 创建全连接层作为解码器
        if bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, num_classes)
    def construct(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings)
        encoding = self.concat((output[0], output[499]))
        outputs = self.decoder(encoding)
        return outputs

## 模型训练

import argparse
import os
import numpy as np
from mindspore import Tensor, nn, context, load_param_into_net, load_checkpoint
from mindspore.train import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor,Accuracy, Model
# 如果条件为真，则执行以下代码块
if 1:
    args = args_train
    cfg = lstm_cfg
# 设置运行模式和设备目标
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)


#数据预处理，生成权重文件
if args.preprocess == "true":
    print("============== Starting Data Pre-processing ==============")
    convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)


#加载嵌入矩阵
embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
# 创建网络
network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)
# 加载预训练模型参数
if args.pre_trained:
    load_param_into_net(network, load_checkpoint(args.pre_trained))
# 定义损失函数、优化器和评价指标
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
loss_cb = LossMonitor()
model = Model(network, loss, opt, {'acc': Accuracy()})
print("============== Starting Training ==============")
# 训练模型
num_steps = 10
ds_train = lstm_create_dataset(args.preprocess_path, cfg.batch_size, 1)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="lstm", directory=args.ckpt_path, config=config_ck)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
# 根据设备目标选择训练方式
if args.device_target == "CPU":
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=False)
else:
    model.train(cfg.num_epochs, ds_train, callbacks=[time_cb, ckpoint_cb, loss_cb])
print("============== Training Success ==============")

## 模型测试

import argparse
import os
import numpy as np
from mindspore import Tensor, nn, context, load_checkpoint, load_param_into_net
from mindspore.train import Accuracy
from mindspore.train import LossMonitor, Model
if 1:
    args = args_test
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=args.device_target)
    if args.preprocess == "true":
        print("============== Starting Data Pre-processing ==============")
        convert_to_mindrecord(cfg.embed_size, args.aclimdb_path, args.preprocess_path, args.glove_path)
    embedding_table = np.loadtxt(os.path.join(args.preprocess_path, "weight.txt")).astype(np.float32)
    network = SentimentNet(vocab_size=embedding_table.shape[0],
                           embed_size=cfg.embed_size,
                           num_hiddens=cfg.num_hiddens,
                           num_layers=cfg.num_layers,
                           bidirectional=cfg.bidirectional,
                           num_classes=cfg.num_classes,
                           weight=Tensor(embedding_table),
                           batch_size=cfg.batch_size)
# 定义损失函数和优化器
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)
# 定义评估指标
    loss_cb = LossMonitor()
    model = Model(network, loss, opt, {'acc': Accuracy()})
    print("============== Starting Testing ==============")
# 创建评估数据集
    ds_eval = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)
    param_dict = load_checkpoint(args.ckpt_path)
# 加载训练好的模型参数
    load_param_into_net(network, param_dict)
# 在设备上进行评估
    if args.device_target == "CPU":
        acc = model.eval(ds_eval, dataset_sink_mode=False)
    else:
        acc = model.eval(ds_eval)
    print("============== {} ==============".format(acc))


