from sklearn.model_selection import StratifiedKFold  

from models import build_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score as sklearn_f1_score, roc_auc_score, average_precision_score
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# 加载数据
train_data = pd.read_csv('processed_train_data_with_binding.csv')

# 转换为 NumPy 数组
X_en = np.array([ast.literal_eval(x) for x in train_data['enhancer_code_padded']])
X_pr = np.array([ast.literal_eval(x) for x in train_data['promoter_code_padded']])
y = train_data['label'].values


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


# 用于保存各次交叉验证的结果
fold_accuracies = []
fold_f1_scores = []
fold_auroc = []
fold_auprc = []

# 用于保存AUROC和AUPRC曲线数据
roc_curve_data = []
pr_curve_data = []

# 用于绘制平均 AUROC 和 AUPRC
mean_fpr = np.linspace(0, 1, 100)
mean_precision = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# 开始十折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X_en, y)):
    print(f"Training fold {fold + 1}...")

    # 划分训练集和验证集
    X_en_train, X_en_val = X_en[train_idx], X_en[val_idx]
    X_pr_train, X_pr_val = X_pr[train_idx], X_pr[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

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
        metrics=["accuracy", tf.keras.metrics.AUC(name="AUROC", curve="ROC"),
                 tf.keras.metrics.AUC(name="AUPRC", curve="PR")]
    )

    # 类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # 回调函数
    checkpoint = ModelCheckpoint(f'best_model_fold{fold + 1}.keras', save_best_only=True, monitor='val_AUPRC',
                                 mode='max')
    early_stopping = EarlyStopping(monitor='val_AUPRC', patience=15, restore_best_weights=True)

    # 训练模型
    model.fit(
        {'enhancer_input': X_en_train, 'promoter_input': X_pr_train}, y_train,
        validation_data=({'enhancer_input': X_en_val, 'promoter_input': X_pr_val}, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stopping]
    )

    # 验证模型
    y_pred_prob = model.predict({'enhancer_input': X_en_val, 'promoter_input': X_pr_val}).flatten()
    y_pred_sklearn = (y_pred_prob > 0.5).astype(int)
    sklearn_f1 = sklearn_f1_score(y_val, y_pred_sklearn)

    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(y_val, y_pred_prob)
    auprc = average_precision_score(y_val, y_pred_prob)

    # 保存本折的 F1-score、AUROC 和 AUPRC
    fold_f1_scores.append(sklearn_f1)
    fold_auroc.append(auroc)
    fold_auprc.append(auprc)

    # 保存本折的 F1-score、AUROC 和 AUPRC
    fold_f1_scores.append(sklearn_f1)
    fold_auroc.append(auroc)
    fold_auprc.append(auprc)

    # 评估模型
    results = model.evaluate({'enhancer_input': X_en_val, 'promoter_input': X_pr_val}, y_val)
    fold_accuracies.append(results[1])  # 保存准确率

    # 用于绘制平均 AUROC 和 AUPRC
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    mean_auc = auroc
    mean_fpr, mean_tpr = mean_fpr, mean_tpr

    precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
    #mean_precision = np.interp(mean_recall, recall, precision)
    mean_precision = np.interp(mean_recall, recall[::-1], precision[::-1])

    # 保存每一折的ROC和PR曲线数据
    roc_curve_data.append((mean_fpr, mean_tpr))
    pr_curve_data.append((mean_recall, mean_precision))

    # 释放内存
    tf.keras.backend.clear_session()

print(f"平均准确率: {np.mean(fold_accuracies)}")
print(f"平均 F1-score: {np.mean(fold_f1_scores)}")
print(f"平均 AUROC: {np.mean(fold_auroc)}")
print(f"平均 AUPRC: {np.mean(fold_auprc)}")

output_dir = 'k26TF-results'  
os.makedirs(output_dir, exist_ok=True)

roc_curve_file = os.path.join(output_dir, 'roc_curve_data.csv')
pr_curve_file = os.path.join(output_dir, 'pr_curve_data.csv')

roc_df = pd.DataFrame(roc_curve_data, columns=['FPR', 'TPR'])
roc_df.to_csv(roc_curve_file, index=False)

pr_df = pd.DataFrame(pr_curve_data, columns=['Recall', 'Precision'])
pr_df.to_csv(pr_curve_file, index=False)

# 绘制平均 AUROC
plt.figure(figsize=(12, 6))
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean AUROC: {np.mean(fold_auroc):.3f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  
plt.title('Mean AUROC (Receiver Operating Characteristic)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 绘制平均 AUPRC
plt.figure(figsize=(12, 6))
plt.plot(mean_recall, mean_precision, color='b', label=f'Mean AUPRC: {np.mean(fold_auprc):.3f}')
plt.title('Mean AUPRC (Precision-Recall Curve)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()
