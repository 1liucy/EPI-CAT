from models import build_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.model_selection import train_test_split  
import tensorflow as tf
import numpy as np
import pandas as pd
import ast


train_data = pd.read_csv('K-train.csv')

X_en = np.array([ast.literal_eval(x) for x in train_data['enhancer_code_padded']])
X_pr = np.array([ast.literal_eval(x) for x in train_data['promoter_code_padded']])
y = train_data['label'].values

# 使用 train_test_split 划分数据集：95% 训练集，5% 验证集
X_en_train, X_en_val, X_pr_train, X_pr_val, y_train, y_val = train_test_split(
    X_en, X_pr, y, test_size=0.05, stratify=y, random_state=250)

# 确保数据格式正确
print("训练集 enhancer 输入形状:", X_en_train.shape)  
print("训练集 promoter 输入形状:", X_pr_train.shape)  
print("训练集标签形状:", y_train.shape)

# 构建模型
model = build_model(
    en_max_len=X_en_train.shape[1],
    pr_max_len=X_pr_train.shape[1],
    onehot_dim=X_en_train.shape[2],  
    embed_dim=64,  
    num_heads=8,
    feed_forward_size=128
)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="AUROC", curve="ROC"),  
        tf.keras.metrics.AUC(name="AUPRC", curve="PR"),   
    ]
)


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weight_dict)

checkpoint = ModelCheckpoint(filepath='model.keras', save_best_only=True, monitor='val_AUPRC', mode='max')
early_stopping = EarlyStopping(monitor='val_AUPRC', patience=15, restore_best_weights=True)

history = model.fit(
    {'enhancer_input': X_en_train, 'promoter_input': X_pr_train}, y_train,
    validation_data=({'enhancer_input': X_en_val, 'promoter_input': X_pr_val}, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stopping]
)

# 验证 Sklearn F1-score
y_pred_sklearn = (model.predict({'enhancer_input': X_en_val, 'promoter_input': X_pr_val}) > 0.5).astype(int)
sklearn_f1 = sklearn_f1_score(y_val, y_pred_sklearn)
print(f"Sklearn F1-score (validation): {sklearn_f1}")

# 保存模型
model.save('model.keras')
print("Model saved as 'model.keras'")
