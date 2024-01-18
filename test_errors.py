import torch
import numpy as np


def compute_errors_torch(gt: torch.Tensor, pred: torch.Tensor):
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


def compute_errors_np(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    # select only the values that are greater than zero
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1 = 0.0
    if a2 != a2:
        a2 = 0.0
    if a3 != a3:
        a3 = 0.0

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    mae = (np.abs(gt - pred)).mean()
    if abs_rel != abs_rel:
        abs_rel = 0.0
    if log_10 != log_10:
        log_10 = 0.0
    if mae != mae:
        mae = 0.0

    return abs_rel, rmse, a1, a2, a3, log_10, mae


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


gt = torch.rand(8, 1, 128, 128)
pred = torch.rand(8, 1, 128, 128)

gt_np = gt.numpy()
pred_np = pred.numpy()

torchre = compute_errors_torch(gt, pred)
nps = compute_errors_np(gt_np, pred_np)
batch = compute_errors_batch(gt, pred)

for i in range(len(torchre)):
    # 保留小数点后四位
    print('torch: {:.4f}, np: {:.4f}, batch: {:.4f}'.format(torchre[i], nps[i], batch[i]))
