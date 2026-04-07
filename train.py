"""
消融实验训练脚本 - 支持4种模型变体
使用固定的 data/train 和 data/val 目录

用法:
    # 快速消融实验（自动减少数据量和轮次）
    python train_ablation.py --model all

    # 完整训练单个模型
    python train_ablation.py --model baseline --full-training --epochs 100

    # 指定消融实验参数
    python train_ablation.py --model baseline cbam --ablation-mode --ablation-epochs 15
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# 导入模型
from models import BaselineUNet, CBAMUNet, DenseASPPUNet, DenseASPPCBAMUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate  # 使用增强版的evaluate函数

# 设置目录
dir_train_img = Path('./data/train/imgs/')
dir_train_mask = Path('./data/train/masks/')
dir_val_img = Path('./data/val/imgs/')
dir_val_mask = Path('./data/val/masks/')
dir_checkpoint = Path('./checkpoints/')


def create_train_val_loaders(batch_size, img_scale, num_workers, ablation_mode=False, ablation_data_ratio=0.3):
    """
    创建训练集和验证集的数据加载器
    使用固定的 data/train 和 data/val 目录，不再随机划分

    Args:
        ablation_mode: 是否为消融实验模式（使用更少数据）
        ablation_data_ratio: 消融实验模式下使用的数据比例
    """
    # 1. 创建训练数据集
    try:
        full_train_dataset = CarvanaDataset(dir_train_img, dir_train_mask, img_scale)
        logging.info("使用 CarvanaDataset 加载训练集")
    except (AssertionError, RuntimeError, IndexError):
        full_train_dataset = BasicDataset(dir_train_img, dir_train_mask, img_scale)
        logging.info("使用 BasicDataset 加载训练集")

    # 2. 创建验证数据集
    try:
        full_val_dataset = CarvanaDataset(dir_val_img, dir_val_mask, img_scale)
        logging.info("使用 CarvanaDataset 加载验证集")
    except (AssertionError, RuntimeError, IndexError):
        full_val_dataset = BasicDataset(dir_val_img, dir_val_mask, img_scale)
        logging.info("使用 BasicDataset 加载验证集")

    # 消融实验模式：减少数据量
    if ablation_mode:
        train_size = int(len(full_train_dataset) * ablation_data_ratio)
        val_size = int(len(full_val_dataset) * ablation_data_ratio)

        # 随机采样数据子集（固定种子确保可重复性）
        train_indices = random.sample(range(len(full_train_dataset)), train_size)
        val_indices = random.sample(range(len(full_val_dataset)), val_size)

        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)

        logging.info(f"消融实验模式: 使用 {ablation_data_ratio*100:.0f}% 的数据")
        logging.info(f'  原始训练集: {len(full_train_dataset)} -> 消融训练集: {len(train_dataset)}')
        logging.info(f'  原始验证集: {len(full_val_dataset)} -> 消融验证集: {len(val_dataset)}')
    else:
        train_dataset = full_train_dataset
        val_dataset = full_val_dataset

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def get_model(model_name, n_channels=3, n_classes=1, bilinear=False, device=None):
    """根据模型名称返回对应的模型"""
    model_map = {
        'baseline': BaselineUNet,
        'cbam': CBAMUNet,
        'denseaspp': DenseASPPUNet,
        'full': DenseASPPCBAMUNet
    }

    if model_name not in model_map:
        raise ValueError(f"未知模型: {model_name}. 请选择: {list(model_map.keys())}")

    model = model_map[model_name](n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

    if device:
        model = model.to(device)
        model = model.to(memory_format=torch.channels_last)

    return model


def train_model(
        model,
        model_name,
        device,
        train_loader,
        val_loader,
        n_train,
        n_val,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_wandb: bool = False,
        ablation_mode: bool = False,
):
    """
    训练单个模型
    """
    # 初始化wandb（可选）
    if use_wandb:
        import wandb
        experiment = wandb.init(project='UNet-Ablation', name=model_name, resume='allow', anonymous='must')
        experiment.config.update(dict(
            model=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            amp=amp,
            ablation_mode=ablation_mode
        ))

    # 设置优化器、损失函数、调度器
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 损失函数
    if model.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_dice': [],
        'val_loss': [],
        'val_iou': [],
        'val_precision': [],
        'val_recall': [],
        'best_metrics': {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'loss': float('inf')
        },
        'best_epoch': 0
    }

    global_step = 0

    logging.info(f'''开始训练: {model_name}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Mixed Precision: {amp}
        Ablation Mode:   {ablation_mode}
    ''')

    # 创建模型保存目录
    model_save_dir = dir_checkpoint / model_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs} [{model_name}]', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                if model.n_classes == 1:
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                else:
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if use_wandb:
                    experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})

        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # 使用增强版的evaluate函数，获取详细指标
        val_dice, val_loss, val_iou, val_precision, val_recall = evaluate(
            model, val_loader, device, amp, return_details=True
        )

        # 更新学习率调度器（基于Dice分数）
        scheduler.step(val_dice)

        # 记录验证指标
        history['val_dice'].append(val_dice)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        # 打印详细日志
        logging.info(f'Epoch {epoch} [{model_name}]:')
        logging.info(f'  Train Loss: {avg_train_loss:.4f}')
        logging.info(f'  Val Loss: {val_loss:.4f}')
        logging.info(f'  Val Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        logging.info(f'  Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

        if use_wandb:
            experiment.log({
                'train_loss_epoch': avg_train_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            })

        # 保存最佳模型（基于Dice分数）
        if val_dice > history['best_metrics']['dice']:
            history['best_metrics'] = {
                'dice': val_dice,
                'iou': val_iou,
                'precision': val_precision,
                'recall': val_recall,
                'loss': val_loss,
                'epoch': epoch
            }
            history['best_epoch'] = epoch

            if save_checkpoint:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metrics': history['best_metrics'],
                    'history': history
                }
                torch.save(checkpoint, model_save_dir / f'{model_name}_best.pth')
                logging.info(f'[{model_name}] 保存最佳模型，Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')

    # 保存最终模型和历史
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_metrics': history['best_metrics'],
        'best_epoch': history['best_epoch']
    }
    torch.save(final_checkpoint, model_save_dir / f'{model_name}_final.pth')

    # 保存历史记录为npy文件
    np.save(model_save_dir / f'{model_name}_history.npy', history)

    # 保存为CSV格式便于分析
    try:
        import pandas as pd
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_dice': history['val_dice'],
            'val_iou': history['val_iou'],
            'val_precision': history['val_precision'],
            'val_recall': history['val_recall']
        })
        metrics_df.to_csv(model_save_dir / f'{model_name}_metrics.csv', index=False)
        logging.info(f'[{model_name}] 训练指标已保存到CSV文件')
    except ImportError:
        logging.warning("pandas未安装，跳过CSV保存")

    if use_wandb:
        experiment.finish()

    logging.info(f'[{model_name}] 训练完成!')
    logging.info(f'  最佳Dice: {history["best_metrics"]["dice"]:.4f} (Epoch {history["best_epoch"]})')
    logging.info(f'  对应IoU: {history["best_metrics"]["iou"]:.4f}')
    logging.info(f'  Precision: {history["best_metrics"]["precision"]:.4f}, Recall: {history["best_metrics"]["recall"]:.4f}')

    return model, history, history['best_metrics']['dice']


def run_ablation_experiments(args):
    """
    运行消融实验 - 训练多个模型
    """
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    # 检查数据目录是否存在
    if not dir_train_img.exists() or not dir_train_mask.exists():
        logging.error(f"训练集目录不存在: {dir_train_img} 或 {dir_train_mask}")
        logging.error("请确保数据放在以下位置:")
        logging.error("  data/train/imgs/  - 训练图像")
        logging.error("  data/train/masks/ - 训练标签")
        logging.error("  data/val/imgs/    - 验证图像")
        logging.error("  data/val/masks/   - 验证标签")
        sys.exit(1)

    if not dir_val_img.exists() or not dir_val_mask.exists():
        logging.error(f"验证集目录不存在: {dir_val_img} 或 {dir_val_mask}")
        sys.exit(1)

    # 判断是否为消融实验模式
    # 当指定--ablation-mode或训练多个模型时自动启用
    ablation_mode = args.ablation_mode or (args.model == 'all' and not args.full_training)

    # 设置训练轮次（消融实验使用较少轮次）
    if ablation_mode:
        if args.ablation_epochs:  # 使用指定的消融轮数
            epochs = args.ablation_epochs
        elif args.epochs == 50:  # 使用默认值时
            epochs = 20  # 消融实验默认20轮
            logging.info(f"消融实验模式: 自动将轮数从 {args.epochs} 调整为 {epochs}")
        else:
            epochs = args.epochs
    else:
        epochs = args.epochs

    # 创建训练集和验证集的数据加载器
    logging.info("加载数据集...")
    train_loader, val_loader, n_train, n_val = create_train_val_loaders(
        batch_size=args.batch_size,
        img_scale=args.scale,
        num_workers=os.cpu_count(),
        ablation_mode=ablation_mode,
        ablation_data_ratio=args.ablation_data_ratio
    )
    logging.info(f'训练集: {n_train} 张图片')
    logging.info(f'验证集: {n_val} 张图片')

    # 确定要训练的模型列表
    if args.model == 'all':
        model_list = ['baseline', 'cbam', 'denseaspp', 'full']
    elif isinstance(args.model, list):
        model_list = args.model
    else:
        model_list = [args.model]

    logging.info(f'消融实验将训练以下模型: {model_list}')

    # 存储所有模型的结果
    all_results = {}

    for model_name in model_list:
        logging.info(f'\n{"="*60}')
        logging.info(f'开始训练模型: {model_name.upper()}')
        logging.info(f'{"="*60}\n')

        # 获取模型
        model = get_model(
            model_name=model_name,
            n_channels=args.channels,
            n_classes=args.classes,
            bilinear=args.bilinear,
            device=device
        )

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'模型参数量: 总计 {total_params:,}, 可训练 {trainable_params:,}')

        # 训练模型
        model, history, best_dice = train_model(
            model=model,
            model_name=model_name,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            n_train=n_train,
            n_val=n_val,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_checkpoint=args.save_checkpoint,
            amp=args.amp,
            use_wandb=args.wandb,
            ablation_mode=ablation_mode
        )

        all_results[model_name] = {
            'best_dice': best_dice,
            'best_metrics': history['best_metrics'],
            'history': history,
            'model': model,
            'params': total_params
        }

    # 打印对比结果
    print_results_summary(all_results)

    return all_results


def print_results_summary(all_results):
    """
    打印消融实验对比结果（包含多种指标）
    """
    print(f'\n{"="*90}')
    print('消融实验对比结果（验证集上最佳指标）')
    print(f'{"="*90}')

    # 创建表格
    print(f'\n{"模型":<15} {"Dice":<8} {"IoU":<8} {"Precision":<10} {"Recall":<8} {"Loss":<8} {"参数量":<12} {"最佳轮数":<8}')
    print(f'{"-"*90}')

    for model_name, results in sorted(all_results.items(), key=lambda x: x[1]['best_dice'], reverse=True):
        metrics = results['best_metrics']
        params = results['params']

        print(f'{model_name:<15} {metrics["dice"]:.4f}   {metrics["iou"]:.4f}   '
              f'{metrics["precision"]:.4f}     {metrics["recall"]:.4f}   '
              f'{metrics["loss"]:.4f}   {params:>10,}   {metrics["epoch"]:<8}')

    print(f'{"="*90}\n')

    # 计算相对于基线的提升
    if 'baseline' in all_results:
        baseline_dice = all_results['baseline']['best_dice']
        baseline_iou = all_results['baseline']['best_metrics']['iou']
        print("相对于基线模型(Baseline)的提升:")
        print(f'{"模型":<15} {"Dice提升":<12} {"IoU提升":<12}')
        print(f'{"-"*40}')

        for model_name, results in all_results.items():
            if model_name != 'baseline':
                dice_improve = (results['best_dice'] - baseline_dice) / baseline_dice * 100
                iou_improve = (results['best_metrics']['iou'] - baseline_iou) / baseline_iou * 100
                print(f'{model_name:<15} +{dice_improve:>6.2f}%        +{iou_improve:>6.2f}%')

    print("\n注意: 以上是在验证集上的最佳指标")
    print("最终论文结果请使用 test_on_testset.py 在测试集上评估")


def get_args():
    parser = argparse.ArgumentParser(description='消融实验 - 训练U-Net变体模型')

    # 模型选择参数
    parser.add_argument('--model', '-m', nargs='+', default='all',
                        choices=['baseline', 'cbam', 'denseaspp', 'full', 'all'],
                        help='选择要训练的模型 (可多选，或使用 all)')

    # 消融实验模式参数
    parser.add_argument('--ablation-mode', action='store_true', default=False,
                        help='启用消融实验模式（使用更少数据和轮次）')
    parser.add_argument('--full-training', action='store_true', default=False,
                        help='完整训练模式（不使用消融实验的优化）')
    parser.add_argument('--ablation-epochs', type=int, default=None,
                        help='消融实验模式的训练轮数（默认20）')
    parser.add_argument('--ablation-data-ratio', type=float, default=0.3,
                        help='消融实验模式使用的数据比例（默认0.3）')

    # 训练参数
    parser.add_argument('--epochs', '-e', type=int, default=50, help='完整训练轮数')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='批次大小')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='学习率', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='图像缩放比例')
    parser.add_argument('--amp', action='store_true', default=False, help='使用混合精度训练')
    parser.add_argument('--bilinear', action='store_true', default=False, help='使用双线性上采样')

    # 数据参数
    parser.add_argument('--channels', '-c', type=int, default=3, help='输入图像通道数')
    parser.add_argument('--classes', '-cl', type=int, default=1, help='分割类别数 (1=二分类)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save-checkpoint', action='store_true', default=True, help='保存检查点')
    parser.add_argument('--wandb', action='store_true', default=False, help='使用wandb记录')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建检查点目录
    dir_checkpoint.mkdir(parents=True, exist_ok=True)

    # 打印实验配置
    logging.info("="*60)
    logging.info("实验配置")
    logging.info("="*60)
    logging.info(f"模型: {args.model}")

    # 判断模式
    ablation_mode = args.ablation_mode or (args.model == 'all' and not args.full_training)
    logging.info(f"消融实验模式: {ablation_mode}")

    if ablation_mode:
        epochs = args.ablation_epochs if args.ablation_epochs else 20
        logging.info(f"  数据比例: {args.ablation_data_ratio}")
        logging.info(f"  训练轮数: {epochs}")
    else:
        logging.info(f"  训练轮数: {args.epochs}")

    logging.info(f"批次大小: {args.batch_size}")
    logging.info(f"学习率: {args.lr}")
    logging.info("="*60 + "\n")

    # 运行消融实验
    results = run_ablation_experiments(args)

    logging.info('所有实验完成!')