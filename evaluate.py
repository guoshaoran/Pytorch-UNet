"""
增强版 evaluate.py - 返回更多评估指标
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, return_details=False):
    """
    评估模型性能

    Args:
        net: 模型
        dataloader: 数据加载器
        device: 设备
        amp: 是否使用混合精度
        return_details: 是否返回详细指标（IoU, Precision, Recall）

    Returns:
        如果 return_details=False: (dice_score, loss)
        如果 return_details=True: (dice_score, loss, iou, precision, recall)
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    val_loss = 0

    # 用于计算详细指标的累积器
    all_preds = []
    all_true = []

    criterion = torch.nn.BCEWithLogitsLoss() if net.n_classes == 1 else torch.nn.CrossEntropyLoss()

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round',
                         unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # 移动到设备
            if net.n_classes == 1:
                mask_true = mask_true.to(device=device, dtype=torch.float32)
            else:
                mask_true = mask_true.to(device=device, dtype=torch.long)

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # 预测
            mask_pred = net(image)

            # 计算损失
            if net.n_classes == 1:
                loss = criterion(mask_pred.squeeze(1), mask_true.float())
            else:
                loss = criterion(mask_pred, mask_true)
            val_loss += loss.item()

            # 计算Dice分数
            if net.n_classes == 1:
                mask_pred_binary = (torch.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred_binary, mask_true, reduce_batch_first=False)

                if return_details:
                    all_preds.append(mask_pred_binary.cpu().numpy().flatten())
                    all_true.append(mask_true.cpu().numpy().flatten())
            else:
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_class = mask_pred.argmax(dim=1)
                mask_pred_onehot = F.one_hot(mask_pred_class, net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred_onehot[:, 1:], mask_true_onehot[:, 1:],
                                                    reduce_batch_first=False)

                if return_details:
                    all_preds.append(mask_pred_class.cpu().numpy().flatten())
                    all_true.append(mask_true.cpu().numpy().flatten())

    # 计算平均值
    avg_dice = dice_score / max(num_val_batches, 1)
    avg_loss = val_loss / max(num_val_batches, 1)

    if not return_details:
        return avg_dice, avg_loss

    # 计算详细指标
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # 计算 IoU, Precision, Recall
    intersection = np.logical_and(all_preds, all_true).sum()
    union = np.logical_or(all_preds, all_true).sum()
    iou = intersection / (union + 1e-7)

    tp = intersection  # True Positive
    fp = np.logical_and(all_preds, ~all_true).sum()
    fn = np.logical_and(~all_preds, all_true).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return avg_dice, avg_loss, iou, precision, recall