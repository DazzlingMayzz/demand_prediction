# 离散制造行业产品物料需求智能预测系统

## 项目简介
本项目是一个基于机器学习的离散制造行业产品物料需求预测系统。系统通过分析历史数据，使用多种机器学习模型进行训练和预测，为制造企业提供未来三个月的物料需求预测。

## 功能特点
1. 数据预处理
   - 自动处理日期格式
   - 特征工程（时间特征、统计特征等）
   - 数据清洗和验证
   - 支持大规模数据处理

2. 模型训练
   - 支持多种机器学习模型：
     * RandomForest（随机森林）
     * XGBoost（极限梯度提升）
     * LightGBM（轻量级梯度提升）
   - 自动模型选择和评估
   - 特征重要性分析
   - 模型性能评估（R2分数、RMSE）

3. 需求预测
   - 支持未来三个月需求预测
   - 预测结果保留两位小数
   - 自动生成预测报告
   - 支持批量预测

## 系统架构
```
src/
├── main.py              # 主程序入口
├── data_preprocessing.py # 数据预处理模块
├── model_training.py    # 模型训练模块
├── model_predict.py     # 预测模块
├── model_evaluation.py  # 模型评估模块
└── utils.py            # 工具函数模块
```

## 环境要求
- Python 3.8+
- 依赖包：
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  scikit-learn>=0.24.2
  lightgbm>=3.2.1
  xgboost>=1.4.2
  matplotlib>=3.4.3
  seaborn>=0.11.2
  tqdm>=4.62.0
  python-dateutil>=2.8.2
  pytz>=2023.3
  joblib>=1.0.1
  ```

## 安装说明
1. 克隆项目到本地
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
1. 数据准备
   - 准备历史需求数据，包含以下字段：
     * 过账日期
     * 工厂编码
     * 物料编码
     * 需求量

2. 运行预测
   ```bash
   python src/main.py
   ```

3. 查看结果
   - 预测结果将保存在 `data/predictions.csv` 文件中
   - 特征重要性分析结果保存在 `data/feature_importance.csv` 文件中

## 模型说明
1. RandomForest 模型
   - 参数配置：
     * n_estimators=200
     * max_depth=15
     * min_samples_split=5
     * min_samples_leaf=2

2. XGBoost 模型
   - 参数配置：
     * n_estimators=200
     * max_depth=7
     * learning_rate=0.1
     * subsample=0.8
     * colsample_bytree=0.8

3. LightGBM 模型
   - 参数配置：
     * n_estimators=100
     * max_depth=5
     * learning_rate=0.2
     * subsample=0.8
     * colsample_bytree=0.8
     * num_leaves=31
     * min_data_in_leaf=20
     * max_bin=255

## 性能指标
- R2分数：衡量模型解释数据的能力
- RMSE：衡量预测误差的大小
- 特征重要性：帮助理解影响需求的关键因素

## 更新日志
- 2024-04-21：优化 LightGBM 训练速度
- 2024-04-18：添加特征重要性分析
- 2024-04-17：实现多模型训练和评估