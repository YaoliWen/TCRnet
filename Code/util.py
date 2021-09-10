import numpy as np
import os
import torch
import shutil
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 矩阵统计
class MatrixMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes)) # 分布矩阵，第一索引为标签，第二索引为预测值
        self.matrix_val = np.zeros((num_classes, num_classes))

    def update(self, score, target):
        self.matrix_val = np.zeros((self.num_classes, self.num_classes))
        prec = score.argmax(axis=1) # B
        for i,j in zip(target, prec):
            self.matrix_val[i][j] += 1
        self.matrix += self.matrix_val
    
    def confus_matrix(self):
        confusion_matrix = (self.matrix.T/(self.matrix.sum(1))).T
        return confusion_matrix
    
    def label_acc(self):
        acc = np.diagonal(self.matrix) / self.matrix.sum(1)
        return acc

    def avg_acc(self):
        acc = np.diagonal(self.matrix).sum() / self.matrix.sum()
        return acc

    def val_acc(self):
        acc = np.diagonal(self.matrix_val).sum() / self.matrix_val.sum()
        return acc

# 数值统计
class AverageMeter(object): 
    def __init__(self):
        self.value = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count
    
    def val(self):
        return self.value

# 输出重定向
class Logger(object): 
    def __init__(self, filename='default.log', stream=sys.stdout):
        os.makedirs(os.path.dirname(filename), mode=0o777, exist_ok=True)
        self.terminal = stream
        self.log = open(filename, 'a', encoding='gb18030')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 储存模型参数
def save_checkpoint(state, is_best, checkp_dir):
    epoch_num = state['epoch']
    full_filename = os.path.join(checkp_dir, 'checkpoint_'+str(epoch_num)+'.pth.tar')
    # save state
    os.makedirs(checkp_dir, mode=0o777, exist_ok=True)
    torch.save(state, full_filename)
    # save best state
    if is_best:
        full_bestname = os.path.join(checkp_dir, 'model_best.pth.tar')
        shutil.copyfile(full_filename, full_bestname)

# 绘制热力图
def heatmap(confusion_matrix, dataset):
    # 指定坐标轴
    if dataset == 'RAF-DB':
        name = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']
    # 制作表格
    df = pd.DataFrame(confusion_matrix, index=name, columns=name)
    # 生成热力图
    sns.set(font_scale=1)
    ax = sns.heatmap(df, annot=True, fmt='.2f', cmap="Blues") # type: axes._subplots.AxesSubplot
    # 设置格式
    plt.xticks(rotation=70)
    plt.rc('font',family='Times New Roman',size=12)
    # 获取Figure对象
    fig = ax.get_figure() # type: figure.Figure
    return fig
    
# 调整学习率
def adjust_learning_rate(args, optimizer, epoch):
    if 0 in [args.lr_interval, args.lr_rate]:
        return
    if epoch >= args.lr_start and (epoch-args.lr_start) % args.lr_interval == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_rate

# base name
def base_file_name(args):
    name = "{dataset}/{subset}/({model})nb_{n_b}-pt_{pt}-nh_{n_h}-d_{dp}-vr{vr}-'{pretrain}'-b_{b}-lr_{lr}-No.{name}".format(
        dataset=args.dataset, subset=args.subset, model=args.model, n_b=args.blocks, pt=args.pool_type, n_h=args.num_heads,
        dp=args.dropout, pretrain=args.pretrain, b=args.batch_size, lr=args.lr, name=args.name, vr= args.var_rate if args.var_loss else '0')
    return name
