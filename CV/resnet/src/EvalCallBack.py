import os, stat
from mindspore import Callback, save_checkpoint


def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    metrics_name = eval_param['metrics_name']
    res = eval_model.eval(eval_ds)
    return res[metrics_name]

# 自定义一个数据收集的回调类EvalCallBack，用于保存精度最高的模型。

class EvalCallBack(Callback):
    """
    回调类，获取训练过程中模型的信息
    """
    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, num_epochs=10,
                 save_best_ckpt=True, ckpt_directory="./", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        self.number_epochs = num_epochs

        if interval < 1:
            raise ValueError("interval should >= 1.")

        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0

        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)

        self.best_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        print('num_epochs: ', num_epochs, ' best_ckpt_path: ', self.best_ckpt_path)
        self.metrics_name = metrics_name

    # 删除ckpt文件
    def remove_ckpoint_file(self, file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)

    # 每一个epoch后，打印训练集的损失值和验证集的模型精度，并保存精度最好的ckpt文件
    '''
    {'epoch_end', 'end'} methods may not be supported in later version, Use methods prefixed 
    with 'on_train' or 'on_eval' instead 
    '''
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss_epoch = cb_params.net_outputs

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)
            print('Epoch {}/{}'.format(cur_epoch, self.number_epochs))
            print('-' * 10)
            print('train Loss: {}'.format(loss_epoch))
            print('val Acc: {}'.format(res))

            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.best_ckpt_path)

    # 训练结束后，打印最好的精度和对应的epoch
    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name, self.best_res, self.best_epoch), flush=True)
