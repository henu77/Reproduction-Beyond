import os

import numpy as np
import torch


def compute_errors(gt: torch.Tensor, pred: torch.Tensor):
    """
    @param gt: 真实深度
    @param pred: 预测深度
    @return:
        [abs_rel: 绝对相对误差,
        rmse: 均方根误差,
        a1: 阈值1,
        a2: 阈值2,
        a3: 阈值3,
        log_10: 对数误差,
        mae: 平均绝对误差]
        计算预测深度与地面真实深度之间的误差指标
    """
    # 选择大于零的数值
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    # 计算阈值
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    # 计算均方根误差
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2).float()).item()

    # 计算绝对相对误差
    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()

    # 计算对数误差
    log_10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred))).item()

    # 计算平均绝对误差
    mae = torch.mean(torch.abs(gt - pred)).item()
    return [abs_rel, rmse, a1, a2, a3, log_10, mae]
def compute_errors_batch(gt: torch.Tensor, pred: torch.Tensor):
    """
    @param gt: 真实深度，大小为[batch_size, 1, height, width]
    @param pred: 预测深度，大小为[batch_size, 1, height, width]
    @return:
        [abs_rel: 绝对相对误差,
        rmse: 均方根误差,
        a1: 阈值1,
        a2: 阈值2,
        a3: 阈值3,
        log_10: 对数误差,
        mae: 平均绝对误差]
        计算预测深度与地面真实深度之间的误差指标
    """
    # 选择大于零的数值
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    # 计算阈值
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    # 计算均方根误差
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2).float()).item()

    # 计算绝对相对误差
    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()

    # 计算对数误差
    log_10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred))).item()

    # 计算平均绝对误差
    mae = torch.mean(torch.abs(gt - pred)).item()
    return [abs_rel, rmse, a1, a2, a3, log_10, mae]


class TextWrite(object):
    ''' Wrting the values to a text file
    '''

    def __init__(self, filename):
        self.filename = filename  # 初始化类，将文件名字符串赋给类的filename属性
        self.file = open(self.filename, "w+")  # 以读写模式打开文件，如果文件不存在就创建它
        self.file.close()  # 关闭文件

    def write_line(self, data_list: list):
        self.file = open(self.filename, "a")  # 以追加模式打开文件
        str_write = ""
        for item in data_list:
            if isinstance(item, int):
                str_write += "{:03d}".format(item)
            if isinstance(item, str):
                str_write += item
            if isinstance(item, float):
                str_write += "{:.6f}".format(item)
            str_write += ","
        str_write += "\n"
        self.file.write(str_write)  # 将str_write的内容写入文件
        self.file.close()  # 关闭文件


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):  # 如果paths是列表而不是字符串
        for path in paths:  # 迭代列表中的每一个项
            mkdir(path)  # 创建对应的目录
    else:
        mkdir(paths)  # 如果paths不是列表，就直接创建目录


def mkdir(path):
    if not os.path.exists(path):  # 如果指定的路径不存在
        os.makedirs(path)  # 就创建目录
