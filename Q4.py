import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("正在执行问题四：阈值自动寻优模型 (强制提升召回率)...\n")

# 1. 读取数据
try:
    df = pd.read_excel('附件.xlsx', sheet_name=1)
except:
    print("❌ 错误：找不到文件附件.xlsx")
    exit()

# 2. 数据准备
cols = df.columns
z_cols = [c for c in cols if 'Z值' in str(c) and ('13' in str(c) or '18' in str(c) or '21' in str(c))]
label_col = cols[27]  # AB列

df_model = df.dropna(subset=z_cols).copy()


def parse_label(val):
    if pd.isna(val): return 0
    if str(val).strip() in ['nan', '', '无', '正常']: return 0
    return 1


df_model['Target'] = df_model[label_col].apply(parse_label)
y_true = df_model['Target']

# 提取三个Z值列 (取绝对值)
Z_matrix = df_model[z_cols].abs().values

# 3. 暴力搜索最佳阈值 (Grid Search)
print(f"正在扫描最佳阈值，目标：召回率 > 90% ...")

best_threshold = 3.0
best_recall = 0
best_acc = 0
found_solution = False

# 从 0.1 到 5.0，步长 0.05 进行扫描
search_range = np.arange(0.1, 5.0, 0.05)
results = []

for t in search_range:
    # 规则：任意一个Z值 > t 就判为异常
    # numpy 的 any(axis=1) 表示只要行内有一个满足条件
    y_pred = (Z_matrix > t).any(axis=1).astype(int)

    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    results.append({'threshold': t, 'recall': rec, 'accuracy': acc})

    # 记录召回率达到 90% 时的最高准确率方案
    if rec >= 0.90:
        if acc > best_acc:  # 找准确率最高的那个
            best_acc = acc
            best_recall = rec
            best_threshold = t
            found_solution = True

# 如果没找到 90% 以上的，就退而求其次找召回率最高的
if not found_solution:
    print("⚠️ 警告：无法达到 90% 召回率，将选择召回率最高的方案。")
    df_res = pd.DataFrame(results)
    best_idx = df_res['recall'].idxmax()
    best_threshold = df_res.loc[best_idx, 'threshold']
    best_recall = df_res.loc[best_idx, 'recall']
    best_acc = df_res.loc[best_idx, 'accuracy']

print(f"\n✅ 寻优完成！")
print(f"最佳 Z 值阈值: > {best_threshold:.2f}")

# 4. 使用最佳阈值生成最终结果
y_pred_final = (Z_matrix > best_threshold).any(axis=1).astype(int)

acc = accuracy_score(y_true, y_pred_final)
rec = recall_score(y_true, y_pred_final)
conf_mat = confusion_matrix(y_true, y_pred_final)
spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

print("\n" + "=" * 30 + " 最终模型评估结果 (请填入论文) " + "=" * 30)
print(f"判定规则: 任意 Z值 > {best_threshold:.2f} 即判为异常")
print(f"1. 准确率 (Accuracy): {acc:.2%}")
print(f"2. 召回率 (Recall):   {rec:.2%} (已最大化)")
print(f"3. 特异性 (Specificity): {spec:.2%}")
print("-" * 60)
print("混淆矩阵:")
print(conf_mat)
print("-" * 60)

# 5. 画图
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
print("✅ 最终版混淆矩阵图已保存为 'Figure6_Auto_Threshold.png'")