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

# Step 3: 构建随机森林模型
model = RandomForestClassifier(
    n_estimators=300,       # 300棵树
    max_depth=15,           # 每棵树最大深度15
    max_features='sqrt',    # 每个节点分裂时使用特征数量
    random_state=42,
    n_jobs=-1               # 多线程加速
)

# Step 4: 模型训练
model.fit(X_train, y_train)

# Step 5: 模型预测与评估
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: 保存模型
joblib.dump(model, 'random_forest_model_600k_params.pkl')
print("模型已保存为 random_forest_model_600k_params.pkl")
