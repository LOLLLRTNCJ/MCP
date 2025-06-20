import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Step 1: 生成模拟数据
X, y = make_classification(
    n_samples=10000,        # 1万条样本
    n_features=40,          # 40个特征
    n_informative=30,       # 30个是有效特征
    n_redundant=5,          # 5个冗余特征
    n_classes=2,            # 二分类
    random_state=42
)

# Step 2: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=1,      
    max_depth=2,        
    max_features='sqrt',    
    random_state=2,
    n_jobs=-1              
)