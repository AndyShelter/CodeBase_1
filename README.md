# YOLOv8 ReID UI

基于YOLOv8和行人重识别(ReID)的实时目标检测与跟踪系统。

## 功能特点

- 支持视频文件、摄像头和RTSP流输入
- 使用YOLOv8进行目标检测
- 使用ReID模型进行行人重识别
- 实时显示检测结果和相似度分数
- 支持自动阈值调整
- 检测结果保存到数据库
- 友好的图形用户界面

## 环境要求

- Python 3.8+
- PyTorch 1.7+
- CUDA 11.0+ (如果使用GPU)
- PyQt5
- OpenCV
- ultralytics
- pymysql

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/yolov8_reid_ui.git
cd yolov8_reid_ui
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备模型权重：
- 将YOLOv8模型权重文件放在 `weight/v8/` 目录下
- 将ReID模型权重文件放在 `weight/strongreid/` 目录下

4. 配置数据库：
- 在MySQL中创建数据库 `reid_detection`
- 修改 `database_manager.py` 中的数据库连接信息

## 使用方法

1. 运行主程序：
```bash
python main.py
```

2. 在界面中：
- 选择目标图片
- 选择视频源（文件/摄像头/RTSP）
- 调整检测参数
- 点击运行开始检测

## 项目结构

```
yolov8_reid_ui/
├── main.py                 # 程序入口
├── main_window.py          # 主窗口类
├── detection_thread.py     # 检测线程类
├── database_manager.py     # 数据库管理类
├── utils/                  # 工具类
├── main_ui/               # UI相关文件
├── weight/                # 模型权重目录
└── config/                # 配置文件目录
```

## 注意事项

- 首次运行时会在 `config` 目录下生成配置文件
- 检测结果默认保存在 `result` 目录
- 确保有足够的磁盘空间存储检测结果

## 许可证

MIT License 



# 行人重识别(ReID)训练与评估系统

这是一个基于PyTorch的行人重识别(ReID)训练与评估系统，专门为DukeMTMC-reID数据集设计。该系统包含了完整的训练、验证和评估流程，并提供了丰富的可视化功能。

## 主要功能

- 支持DukeMTMC-reID数据集的训练和评估
- 实现了遮挡感知的行人重识别模型
- 提供了多种损失函数和优化器选择
- 包含完整的训练过程监控和可视化
- 支持模型性能评估和比较

## 系统架构

### 1. 配置系统
- 使用`Config`类管理所有配置参数
- 支持模型、优化器、数据加载器、数据集等多方面配置
- 可灵活调整训练和测试参数

### 2. 模型结构
- 基于ResNet50的基线模型
- 支持遮挡感知模块
- 可配置的瓶颈层和特征提取策略

### 3. 损失函数
- 交叉熵损失（支持标签平滑）
- 遮挡感知损失
- 可扩展的损失函数系统

### 4. 数据加载
- 支持自定义数据增强
- 灵活的数据集配置
- 高效的数据加载器

### 5. 评估系统
- 支持mAP和CMC评估指标
- 提供详细的性能分析
- 生成可视化报告

## 使用方法

### 1. 环境配置
```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install opencv-python pillow
```

### 2. 数据准备
- 下载DukeMTMC-reID数据集
- 将数据集放在指定目录下
- 确保数据目录结构正确

### 3. 训练模型
```python
python kaggle.py --mode train
```

### 4. 评估模型
```python
python kaggle.py --mode eval
```

### 5. 可视化结果
```python
python kaggle.py --mode visualize
```

## 配置说明

### 模型配置
```python
MODEL = {
    'NUM_CLASSES': 702,  # 类别数
    'LAST_STRIDE': 1,    # 最后一层步长
    'PRETRAIN_PATH': 'path/to/pretrained/model',
    'NAME': 'baseline',
    'PRETRAIN_CHOICE': 'imagenet',
    'NECK': 'bnneck',
    'NECK_FEAT': 'after',
    'IF_LABELSMOOTH': 'on',
    'COS_LAYER': 'no'
}
```

### 优化器配置
```python
SOLVER = {
    'OPTIMIZER_NAME': 'Adam',
    'BASE_LR': 3e-4,
    'WEIGHT_DECAY': 5e-4,
    'STEPS': [30, 55, 80],
    'GAMMA': 0.1,
    'MAX_EPOCHS': 120,
    'IMS_PER_BATCH': 64
}
```

## 可视化功能

1. 训练曲线
   - 损失函数变化
   - 准确率变化
   - 学习率变化

2. 性能评估
   - CMC曲线
   - mAP指标
   - 混淆矩阵

3. 注意力图
   - 特征可视化
   - 遮挡区域检测
   - 热力图显示

## 注意事项

1. 数据准备
   - 确保数据集路径正确
   - 检查数据格式和标签
   - 预处理图像大小

2. 训练过程
   - 监控GPU内存使用
   - 定期保存检查点
   - 调整批量大小

3. 评估过程
   - 使用验证集评估
   - 保存评估结果
   - 生成报告

## 常见问题

1. 内存不足
   - 减小批量大小
   - 使用梯度累积
   - 启用混合精度训练

2. 训练不稳定
   - 调整学习率
   - 使用学习率预热
   - 检查数据增强

3. 性能问题
   - 优化数据加载
   - 使用多GPU训练
   - 启用cudnn基准测试

## 贡献指南

欢迎提交问题和改进建议。请遵循以下步骤：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License 
