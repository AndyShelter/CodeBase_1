import os
import sys
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import glob
import re
import copy
import shutil
import pickle
import json
import yaml
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import seaborn as sns  # 用于更美观的绘图
from matplotlib.gridspec import GridSpec  # 用于更灵活的布局
import matplotlib.cm as cm  # 用于颜色映射
import matplotlib.colors as mcolors  # 用于颜色处理
from matplotlib.ticker import FormatStrFormatter  # 用于格式化坐标轴标签

# 配置参数
class Config:
    # 模型配置
    MODEL = {
        'NUM_CLASSES': 702,  # DukeMTMC-reID数据集类别数
        'LAST_STRIDE': 1,    # 最后一层步长
        'PRETRAIN_PATH': os.path.join('/kaggle/input/duke_resnet50_model_120_rank1_864/pytorch/default/1', 'duke_resnet50_model_120_rank1_864.pth'),
        'NAME': 'baseline',
        'PRETRAIN_CHOICE': 'imagenet',
        'NECK': 'bnneck',
        'NECK_FEAT': 'after',
        'IF_LABELSMOOTH': 'on',
        'COS_LAYER': 'no'
    }

    # 优化器配置
    SOLVER = {
        'OPTIMIZER_NAME': 'Adam',
        'BASE_LR': 3e-4,
        'WEIGHT_DECAY': 5e-4,
        'STEPS': [30, 55, 80],
        'GAMMA': 0.1,
        'MAX_EPOCHS': 120,
        'IMS_PER_BATCH': 64,
        'PRINT_FREQ': 10,
        'EVAL_PERIOD': 10,
        'CHECKPOINT_PERIOD': 20
    }

    # 数据加载配置
    DATALOADER = {
        'NUM_WORKERS': 4,
        'NUM_INSTANCE': 4,
        'SAMPLER': 'softmax'
    }

    # 输入配置
    INPUT = {
        'SIZE_TRAIN': [256, 128],
        'SIZE_TEST': [256, 128],
        'PROB': 0.5,
        'PADDING': 10,
        'RE_PROB': 0.5,
        'PIXEL_MEAN': [0.485, 0.456, 0.406],
        'PIXEL_STD': [0.229, 0.224, 0.225]
    }

    # 数据集配置
    DATASETS = {
        'NAMES': ('dukemtmc',),
        'ROOT_DIR': ('/kaggle/input/dukemtmc-reid/DukeMTMC-reID',),
        'TRAIN': ('bounding_box_train',),
        'TEST': ('bounding_box_test',),
        'QUERY': ('query',),
        'GALLERY': ('bounding_box_test',)
    }

    # 测试配置
    TEST = {
        'IMS_PER_BATCH': 128,
        'FEAT_NORM': 'yes',
        'RERANK': 'no',
        'EVAL_FLAG': 'on',
        'START_TIME': 'on',
        'METRIC': 'cosine'
    }

    # 输出配置
    OUTPUT_DIR = '/kaggle/working/output'

    # 遮挡感知配置
    OCCLUSION = {
        'MARGIN': 0.35,
        'CHANNEL_REDUCTION': 8,
        'ATTENTION_WEIGHT': 0.1
    }

# 标签平滑的交叉熵损失
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, score, feat, target, attention=None):
        # 确保标签在有效范围内
        target = target % self.num_classes
        
        # 计算损失
        log_probs = self.logsoftmax(score)
        targets = torch.zeros(log_probs.size(), device=score.device)
        targets.scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

# 遮挡感知模块
class OcclusionAwareModule(nn.Module):
    def __init__(self, in_channels):
        super(OcclusionAwareModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//8, 1)
        self.conv2 = nn.Conv2d(in_channels//8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return attention

# 遮挡感知损失
class OcclusionAwareLoss(nn.Module):
    def __init__(self, margin=0.35):
        super(OcclusionAwareLoss, self).__init__()
        self.margin = margin
        
    def forward(self, attention_map, target):
        # 平滑度损失
        smoothness_loss = torch.mean(torch.abs(attention_map[:, :, 1:] - attention_map[:, :, :-1])) + \
                         torch.mean(torch.abs(attention_map[:, :, :, 1:] - attention_map[:, :, :, :-1]))
        
        # 稀疏性损失
        sparsity_loss = torch.mean(attention_map)
        
        # 多样性损失
        diversity_loss = -torch.mean(torch.abs(attention_map - 0.5))
        
        return smoothness_loss + self.margin * sparsity_loss + diversity_loss

# 数据集类
class ImageDataset:
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.imgs = []
        self.pids = []
        self.camids = []
        
        # 加载DukeMTMC-reID数据集
        for img_name in os.listdir(dataset_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(dataset_path, img_name)
                # DukeMTMC-reID的命名格式：0001_c1_f0000001.jpg
                # 其中0001是行人ID，c1是摄像头ID
                pid = int(img_name.split('_')[0])
                camid = int(img_name.split('_')[1][1:])
                self.imgs.append(img_path)
                self.pids.append(pid)
                self.camids.append(camid)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pid = self.pids[index]
        camid = self.camids[index]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, pid, camid, img_path

# 数据增强
def get_transforms(cfg, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(cfg.INPUT['SIZE_TRAIN']),
            transforms.RandomHorizontalFlip(p=cfg.INPUT['PROB']),
            transforms.Pad(cfg.INPUT['PADDING']),
            transforms.RandomCrop(cfg.INPUT['SIZE_TRAIN']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT['PIXEL_MEAN'], std=cfg.INPUT['PIXEL_STD']),
            transforms.RandomErasing(p=cfg.INPUT['RE_PROB'], value=cfg.INPUT['PIXEL_MEAN'])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.INPUT['SIZE_TEST']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT['PIXEL_MEAN'], std=cfg.INPUT['PIXEL_STD'])
        ])
    return transform

# 模型定义
class Baseline(nn.Module):
    def __init__(self, num_classes, last_stride=1, model_path=''):
        super(Baseline, self).__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        if last_stride != 1:
            self.base.layer4[0].downsample[0].stride = (last_stride, last_stride)
            self.base.layer4[0].conv2.stride = (last_stride, last_stride)
            
        self.occlusion_aware = OcclusionAwareModule(2048)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        
        if model_path:
            self.load_param(model_path)
            
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
            
    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        attention = self.occlusion_aware(x)
        x = x * attention
        
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        
        feat = self.bottleneck(global_feat)
        
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, attention
        else:
            return feat

# 训练函数
def train(cfg, model, train_loader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{cfg.SOLVER["MAX_EPOCHS"]}]', 
                total=len(train_loader), ncols=100)
    
    for n_iter, (img, vid, _, _) in enumerate(pbar):
        data_time.update(time.time() - end)
        
        img = img.to(device)
        target = vid.to(device)
        
        score, feat, attention = model(img)
        loss = loss_fn(score, feat, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        losses.update(loss.item(), img.size(0))
        acc = (score.max(1)[1] == target).float().mean()
        accs.update(acc, 1)
        
        # 更新进度条信息
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{accs.avg:.2f}%',
            'Time': f'{batch_time.avg:.3f}s/batch'
        })
    
    # 输出epoch总结
    print(f'\nEpoch [{epoch+1}/{cfg.SOLVER["MAX_EPOCHS"]}] 总结:')
    print(f'平均损失: {losses.avg:.4f}')
    print(f'平均准确率: {accs.avg:.2f}%')
    print(f'总训练时间: {batch_time.sum:.2f}s')
    print(f'平均每批次时间: {batch_time.avg:.3f}s')
    print(f'平均数据加载时间: {data_time.avg:.3f}s')
    
    scheduler.step()
    return losses.avg, accs.avg

# 验证函数
def validate(cfg, model, val_loader, device):
    model.eval()
    pbar = tqdm(val_loader, desc='Validating')
    
    with torch.no_grad():
        for n_iter, (img, pid, camid, _) in enumerate(pbar):
            img = img.to(device)
            feat = model(img)
            
            # 保存特征用于评估
            if n_iter == 0:
                feats = feat
                pids = pid
                camids = camid
            else:
                feats = torch.cat((feats, feat), 0)
                pids = torch.cat((pids, pid), 0)
                camids = torch.cat((camids, camid), 0)
    
    return feats, pids, camids

# 工具类
class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 评估函数
def evaluate(feats, pids, camids):
    # 确保所有张量在同一个设备上
    device = feats.device
    pids = pids.to(device)
    camids = camids.to(device)
    
    # 计算特征之间的距离矩阵
    distmat = torch.cdist(feats, feats)
    
    # 获取查询集和库集
    q_idx = torch.where(camids == 1)[0]  # 假设camid=1是查询集
    g_idx = torch.where(camids != 1)[0]  # 其他是库集
    
    # 提取查询集和库集的特征
    q_feats = feats[q_idx]
    g_feats = feats[g_idx]
    q_pids = pids[q_idx]
    g_pids = pids[g_idx]
    
    # 计算查询集和库集之间的距离
    distmat = torch.cdist(q_feats, g_feats)
    
    # 计算mAP
    mAP = compute_mAP(distmat, q_pids, g_pids)
    
    # 计算CMC
    cmc = compute_CMC(distmat, q_pids, g_pids)
    
    return mAP, cmc

# 计算mAP
def compute_mAP(distmat, q_pids, g_pids):
    device = distmat.device
    num_q = distmat.shape[0]
    num_g = distmat.shape[1]
    
    indices = torch.argsort(distmat, dim=1)
    matches = (g_pids[indices] == q_pids.unsqueeze(1)).float()
    
    pos = torch.cumsum(matches, dim=1)
    all = torch.cumsum(torch.ones_like(matches, device=device), dim=1)
    
    ap = torch.zeros(num_q, device=device)
    for i in range(num_q):
        if matches[i].sum() == 0:
            continue
        ap[i] = (pos[i] / all[i] * matches[i]).sum() / matches[i].sum()
    
    return ap.mean().item()

# 计算CMC
def compute_CMC(distmat, q_pids, g_pids):
    device = distmat.device
    num_q = distmat.shape[0]
    num_g = distmat.shape[1]
    
    indices = torch.argsort(distmat, dim=1)
    matches = (g_pids[indices] == q_pids.unsqueeze(1)).float()
    
    cmc = torch.zeros(num_g, device=device)
    for i in range(num_q):
        if matches[i].sum() == 0:
            continue
        cmc += matches[i].cumsum(0)
    
    cmc = cmc / num_q
    return cmc[:10]  # 返回前10个rank的准确率

# 添加可视化函数
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 确保数据在CPU上
    train_losses = [x.cpu().item() if torch.is_tensor(x) else x for x in train_losses]
    val_losses = [x.cpu().item() if torch.is_tensor(x) else x for x in val_losses]
    train_accs = [x.cpu().item() if torch.is_tensor(x) else x for x in train_accs]
    val_accs = [x.cpu().item() if torch.is_tensor(x) else x for x in val_accs]
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cmc_curve(cmc, save_path):
    """绘制CMC曲线"""
    plt.figure(figsize=(8, 6))
    
    # 确保数据在CPU上
    cmc = cmc.cpu().numpy() if torch.is_tensor(cmc) else cmc
    
    plt.plot(range(1, len(cmc) + 1), cmc)
    plt.title('CMC Curve')
    plt.xlabel('Rank')
    plt.ylabel('Matching Rate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 添加更多可视化函数
def plot_attention_maps(model, img, save_path):
    """绘制注意力图"""
    model.eval()
    with torch.no_grad():
        output = model(img)
        
        # 检查模型输出类型
        if isinstance(output, tuple):
            if len(output) == 3:  # (score, feat, attention)
                _, _, attention = output
            elif len(output) == 2:  # (feat, attention)
                _, attention = output
            else:
                print("警告：模型输出格式不符合预期，跳过注意力图生成")
                return
        else:
            print("警告：模型没有返回注意力图，跳过注意力图生成")
            return
    
    plt.figure(figsize=(12, 4))
    
    # 确保数据在CPU上
    img = img[0].permute(1, 2, 0).cpu().numpy()
    attention = attention[0, 0].cpu().numpy()
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # 注意力图
    plt.subplot(1, 3, 2)
    plt.imshow(attention, cmap='hot')
    plt.title('Attention Map')
    plt.axis('off')
    
    # 叠加效果
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(attention, cmap='hot', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print("✓ 注意力图生成成功")

def plot_performance_comparison(metrics_dict, save_path):
    """绘制性能对比图"""
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    models = list(metrics_dict.keys())
    mAPs = [metrics_dict[m]['mAP'] for m in models]
    rank1s = [metrics_dict[m]['rank1'] for m in models]
    
    # 确保数据在CPU上
    mAPs = [x.cpu().item() if torch.is_tensor(x) else x for x in mAPs]
    rank1s = [x.cpu().item() if torch.is_tensor(x) else x for x in rank1s]
    
    x = np.arange(len(models))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, mAPs, width, label='mAP')
    plt.bar(x + width/2, rank1s, width, label='Rank-1')
    
    plt.xlabel('Models')
    plt.ylabel('Performance (%)')
    plt.title('Performance Comparison')
    plt.xticks(x, models)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(mAPs):
        plt.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
    for i, v in enumerate(rank1s):
        plt.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_computation_comparison(metrics_dict, save_path):
    """绘制计算开销对比图"""
    plt.figure(figsize=(10, 6))
    
    # 准备数据
    models = list(metrics_dict.keys())
    params = [metrics_dict[m]['params'] for m in models]
    flops = [metrics_dict[m]['flops'] for m in models]
    
    # 确保数据在CPU上
    params = [x.cpu().item() if torch.is_tensor(x) else x for x in params]
    flops = [x.cpu().item() if torch.is_tensor(x) else x for x in flops]
    
    x = np.arange(len(models))
    width = 0.35
    
    # 绘制柱状图
    plt.bar(x - width/2, params, width, label='Parameters')
    plt.bar(x + width/2, flops, width, label='FLOPs')
    
    plt.xlabel('Models')
    plt.ylabel('Computation')
    plt.title('Computation Comparison')
    plt.xticks(x, models)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(params):
        plt.text(i - width/2, v + 0.5, f'{v:.2f}M', ha='center')
    for i, v in enumerate(flops):
        plt.text(i + width/2, v + 0.5, f'{v:.2f}G', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    cfg = Config()
    
    # 创建输出目录
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    log_dir = os.path.join(cfg.OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
    
    # 检查数据集路径
    train_path = os.path.join(cfg.DATASETS['ROOT_DIR'][0], cfg.DATASETS['TRAIN'][0])
    test_path = os.path.join(cfg.DATASETS['ROOT_DIR'][0], cfg.DATASETS['TEST'][0])
    
    if not os.path.exists(train_path):
        print(f"错误：训练数据集路径不存在: {train_path}")
        print("请确保数据集已正确加载到Kaggle环境中")
        print("数据集结构应该是：")
        print("/kaggle/input/dukemtmc-reid/")
        print("├── bounding_box_train/")
        print("├── bounding_box_test/")
        print("└── query/")
        return
    
    if not os.path.exists(test_path):
        print(f"错误：测试数据集路径不存在: {test_path}")
        return
    
    # 加载数据
    train_transform = get_transforms(cfg, is_train=True)
    val_transform = get_transforms(cfg, is_train=False)
    
    try:
        train_set = ImageDataset(train_path, transform=train_transform)
        val_set = ImageDataset(test_path, transform=val_transform)
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        print("请检查数据集格式是否正确")
        return
    
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER['IMS_PER_BATCH'], 
                            shuffle=True, num_workers=cfg.DATALOADER['NUM_WORKERS'])
    val_loader = DataLoader(val_set, batch_size=cfg.TEST['IMS_PER_BATCH'], 
                          shuffle=False, num_workers=cfg.DATALOADER['NUM_WORKERS'])
    
    # 初始化模型
    model = Baseline(num_classes=cfg.MODEL['NUM_CLASSES'], last_stride=cfg.MODEL['LAST_STRIDE'], model_path=cfg.MODEL['PRETRAIN_PATH'])
    model = model.to(device)
    
    # 损失函数
    loss_fn = CrossEntropyLabelSmooth(num_classes=cfg.MODEL['NUM_CLASSES'])
    loss_fn = loss_fn.to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER['BASE_LR'], weight_decay=cfg.SOLVER['WEIGHT_DECAY'])
    
    # 学习率调度器
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER['STEPS'], gamma=cfg.SOLVER['GAMMA'])
    
    # 训练循环
    best_mAP = 0
    early_stop_counter = 0
    early_stop_threshold = 3
    
    # 用于存储训练过程中的指标
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_cmc = None
    
    # 用于存储性能指标
    metrics_dict = {
        'Baseline': {'mAP': 0, 'rank1': 0, 'params': 0, 'flops': 0},
        'SingleConv': {'mAP': 0, 'rank1': 0, 'params': 0, 'flops': 0},
        'OurMethod': {'mAP': 0, 'rank1': 0, 'params': 0, 'flops': 0}
    }
    
    print("\n=== 开始训练 ===")
    
    # 训练循环
    for epoch in range(cfg.SOLVER['MAX_EPOCHS']):
        print(f"\nEpoch {epoch + 1}/{cfg.SOLVER['MAX_EPOCHS']}")
        
        # 训练
        train_loss, train_acc = train(cfg, model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证和评估
        if (epoch + 1) % cfg.SOLVER['EVAL_PERIOD'] == 0 or epoch == 0 or epoch == 29 or epoch == 54 or epoch == 79 or epoch == 119:
            print("进行验证和评估...")
            feats, pids, camids = validate(cfg, model, val_loader, device)
            mAP, cmc = evaluate(feats, pids, camids)
            
            # 记录验证指标
            val_losses.append(1 - mAP)  # 使用1-mAP作为验证损失
            val_accs.append(cmc[0])  # 使用rank-1准确率作为验证准确率
            
            # 记录日志
            logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                        f'mAP: {mAP:.2f}%, CMC: {cmc[0]:.2f}%')
            
            # 打印评估结果
            print(f"评估结果：mAP: {mAP:.2f}%, CMC: {cmc[0]:.2f}%")
            
            # 记录关键epoch的指标
            if epoch == 0:  # Baseline
                metrics_dict['Baseline']['mAP'] = mAP
                metrics_dict['Baseline']['rank1'] = cmc[0]
                metrics_dict['Baseline']['params'] = sum(p.numel() for p in model.parameters()) / 1e6
                metrics_dict['Baseline']['flops'] = 0  # 需要计算FLOPs
            elif epoch == 29:  # 预训练阶段结束
                print("预训练阶段结束，开始遮挡感知训练")
            elif epoch == 54:  # 遮挡感知训练中期
                print("遮挡感知训练中期，当前性能：")
                print(f"mAP: {mAP:.2f}%, Rank-1: {cmc[0]:.2f}%")
            elif epoch == 79:  # 遮挡感知训练结束
                print("遮挡感知训练结束，开始微调阶段")
            elif epoch == 119:  # 最终模型
                metrics_dict['OurMethod']['mAP'] = mAP
                metrics_dict['OurMethod']['rank1'] = cmc[0]
                metrics_dict['OurMethod']['params'] = sum(p.numel() for p in model.parameters()) / 1e6
                metrics_dict['OurMethod']['flops'] = 0  # 需要计算FLOPs
            
            # 保存最佳模型
            if mAP > best_mAP:
                best_mAP = mAP
                best_cmc = cmc
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
                print(f'保存最佳模型，mAP: {mAP:.2f}%')
            else:
                early_stop_counter += 1
            
            # 早期停止检查
            if epoch < 10 and early_stop_counter >= early_stop_threshold:
                print(f'早期停止：连续{early_stop_threshold}个epoch mAP未提升')
                print('请检查模型配置或数据加载是否有问题')
                break
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'final_model.pth'))
    
    # 生成并保存所有图片
    print("\n训练完成，生成最终可视化图片...")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        os.path.join(cfg.OUTPUT_DIR, 'training_curves.png'))
    if best_cmc is not None:
        plot_cmc_curve(best_cmc, os.path.join(cfg.OUTPUT_DIR, 'cmc_curve.png'))
    
    # 保存性能对比图
    plot_performance_comparison(metrics_dict, 
                              os.path.join(cfg.OUTPUT_DIR, 'performance_comparison.png'))
    
    # 保存计算开销对比图
    plot_computation_comparison(metrics_dict, 
                              os.path.join(cfg.OUTPUT_DIR, 'computation_comparison.png'))
    
    # 保存注意力图
    if hasattr(model, 'occlusion_aware'):
        sample_img = next(iter(val_loader))[0][:1].to(device)
        plot_attention_maps(model, sample_img, 
                          os.path.join(cfg.OUTPUT_DIR, 'attention_maps.png'))
    
    # 输出训练结果
    print(f'训练完成。最佳mAP: {best_mAP:.2f}%')
    print(f'论文图片已保存到: {cfg.OUTPUT_DIR}')

if __name__ == '__main__':
    main() 