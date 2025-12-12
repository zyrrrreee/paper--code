import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel('附件.xlsx', sheet_name=1)
except:
    exit()

cols = df.columns
z_cols = [c for c in cols if 'Z值' in str(c) and ('13' in str(c) or '18' in str(c) or '21' in str(c))]
label_col = cols[27]

df_model = df.dropna(subset=z_cols).copy()

def parse_label(val):
    if pd.isna(val): return 0
    if str(val).strip() in ['nan', '', '无', '正常']: return 0
    return 1


df_model['Target'] = df_model[label_col].apply(parse_label)
y_true = df_model['Target']

Z_matrix = df_model[z_cols].abs().values

best_threshold = 3.0
best_recall = 0
best_acc = 0
found_solution = False

search_range = np.arange(0.1, 5.0, 0.05)
results = []

for t in search_range:
    y_pred = (Z_matrix > t).any(axis=1).astype(int)

    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    results.append({'threshold': t, 'recall': rec, 'accuracy': acc})

    if rec >= 0.90:
        if acc > best_acc:
            best_acc = acc
            best_recall = rec
            best_threshold = t
            found_solution = True

if not found_solution:
    df_res = pd.DataFrame(results)
    best_idx = df_res['recall'].idxmax()
    best_threshold = df_res.loc[best_idx, 'threshold']
    best_recall = df_res.loc[best_idx, 'recall']
    best_acc = df_res.loc[best_idx, 'accuracy']

y_pred_final = (Z_matrix > best_threshold).any(axis=1).astype(int)

acc = accuracy_score(y_true, y_pred_final)
rec = recall_score(y_true, y_pred_final)
conf_mat = confusion_matrix(y_true, y_pred_final)
spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

print(f"准确率: {acc:.2%}")
print(f"召回率: {rec:.2%}")
print(f"特异性: {spec:.2%}")

plt.figure(figsize=(8, 6), dpi=150)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds',
            xticklabels=['预测正常', '预测异常'],
            yticklabels=['实际正常', '实际异常'],
            annot_kws={"size": 16})
plt.title(f'图6：最优阈值判定混淆矩阵\n(Threshold={best_threshold:.2f}, Recall={rec:.1%})', fontsize=14)
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.savefig('Figure6_Auto_Threshold.png')