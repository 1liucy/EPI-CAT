from tensorflow.keras.models import load_model
from transformer import Transformer_Merged_CrossAttention, LayerNormalization, PositionEncoding  
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score as sklearn_f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@tf.keras.utils.register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

test_data = pd.read_csv('K-test.csv')
X_en_test = np.array([np.array(eval(x)) for x in test_data['enhancer_code_padded']])
X_pr_test = np.array([np.array(eval(x)) for x in test_data['promoter_code_padded']])
y_test = test_data['label'].values

# 加载模型
model = load_model(
    'model.keras',
    custom_objects={
        'Transformer_Merged_CrossAttention': Transformer_Merged_CrossAttention,  
        'LayerNormalization': LayerNormalization,
        'PositionEncoding': PositionEncoding,
        'f1_score': f1_score
    }
)

# 模型评估
results = model.evaluate({'enhancer_input': X_en_test, 'promoter_input': X_pr_test}, y_test)
# loss, accuracy, test_auc, test_auprc, test_f1_score = results
loss, accuracy, test_auc, test_auprc = results
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
print(f"Test AUROC: {test_auc}")
print(f"Test AUPRC: {test_auprc}")
#print(f"Test F1-score: {test_f1_score}")

y_pred_prob = model.predict({'enhancer_input': X_en_test, 'promoter_input': X_pr_test}).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

sklearn_f1 = sklearn_f1_score(y_test, y_pred)
print(f"Sklearn F1-score (test set): {sklearn_f1}")

output = pd.DataFrame({
    'enhancer_code': test_data['enhancer_code'],
    'promoter_code': test_data['promoter_code'],
    'true_label': y_test,
    'predicted_label': y_pred,
    'predicted_prob': y_pred_prob
})
output.to_csv('test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'.")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUROC: {test_auc:.3f}")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f"AUPRC: {test_auprc:.3f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()
