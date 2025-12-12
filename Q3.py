import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

print("正在执行问题三：多因素决策树分组（高清优化版）...\n")

# 1. 读取数据
try:
    df = pd.read_excel('附件.xlsx', sheet_name=0)  # 男胎
except Exception as e:
    print(f"❌ 错误：找不到文件 '附件.xlsx'。")
    exit()

# 2. 数据清洗 (与之前保持一致)
col_map = {}
cols = df.columns
for c in cols:
    if '年龄' in str(c) or 'Age' in str(c):
        col_map[c] = 'Age'
    elif '身高' in str(c) or 'Height' in str(c):
        col_map[c] = 'Height'
    elif '体重' in str(c) and '指数' not in str(c) or 'Weight' in str(c):
        col_map[c] = 'Weight'
    elif 'BMI' in str(c):
        col_map[c] = 'BMI'
    elif '孕周' in str(c) or 'Week' in str(c):
        col_map[c] = 'Week'
    elif ('Y' in str(c) and ('浓度' in str(c) or '比例' in str(c)) and 'Z' not in str(c)) or 'Y_Conc' in str(c):
        col_map[c] = 'Y_Conc'

df = df.rename(columns=col_map)

# 提取数值
if df['Week'].dtype == object:
    df['Week_Num'] = df['Week'].astype(str).str.extract(r'(\d+)').astype(float)
else:
    df['Week_Num'] = df['Week']

for c in ['Age', 'Height', 'Weight', 'BMI', 'Y_Conc']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['Age', 'Height', 'Weight', 'BMI', 'Y_Conc', 'Week_Num'])

# 修复单位
if df['Y_Conc'].median() < 1:
    df['Y_Conc'] = df['Y_Conc'] * 100
df = df[df['Y_Conc'] > 0]

# 3. 构建决策树模型
# 特征：BMI, Age, Height, Weight
feature_cols = ['BMI', 'Age', 'Height', 'Weight']
X = df[feature_cols]
y = df['Y_Conc']

# 【优化点1】：增加 min_samples_leaf 防止分太细，限制 max_depth=3 保证图能看清
tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
tree_model.fit(X, y)

# 4. 可视化决策树 (解决问题①：图片看不清)
# figsize 设置为 (25, 12) 超大画布，fontsize 设置为 14
plt.figure(figsize=(25, 12), dpi=300)  # dpi=300 保证高清
plot_tree(tree_model,
          feature_names=feature_cols,
          filled=True,
          rounded=True,
          fontsize=14,  # 字体变大
          precision=1,  # 小数点保留1位
          impurity=False)  # 不显示 impurity (mse)，让图更简洁
plt.title('图5：基于多生理特征的 NIPT 浓度预测决策树', fontsize=20)
plt.savefig('Figure5_Tree_HD.png', dpi=300, bbox_inches='tight')
print("✅ 高清决策树结构图已保存为 'Figure5_Tree_HD.png'")

# 5. 提取规则并计算最佳时点 (解决问题③：描述重复)
df['Leaf_Node'] = tree_model.apply(X)

print("\n" + "=" * 30 + " 自动分组与最佳时点建议 (真实数据版) " + "=" * 30)
# 调整列宽以适应新描述
print(f"{'组别':<4} | {'人群特征描述 (精确范围)':<45} | {'样本数':<6} | {'建议检测周数':<12}")
print("-" * 100)


def analyze_risk_with_error(sub_df):
    stats = sub_df.groupby('Week_Num')['Y_Conc'].apply(lambda x: (x >= 4.0).mean())
    counts = sub_df.groupby('Week_Num')['Y_Conc'].count()
    stats_df = pd.DataFrame({'Pass_Rate': stats, 'Count': counts})
    stats_df = stats_df[stats_df['Count'] >= 3]

    qualified = stats_df[stats_df['Pass_Rate'] >= 0.9]
    if not qualified.empty:
        return int(qualified.index.min()), stats_df['Pass_Rate'].max()
    else:
        if not stats_df.empty:
            return int(stats_df['Pass_Rate'].idxmax()), stats_df['Pass_Rate'].max()
        return -1, 0


# 按预测出的浓度均值排序，均值越低越危险，排在后面
leaf_means = df.groupby('Leaf_Node')['Y_Conc'].mean().sort_values(ascending=False)

for i, leaf in enumerate(leaf_means.index):
    sub = df[df['Leaf_Node'] == leaf]

    # 【优化点2】：不再使用模糊的 if-else，直接计算该组的真实范围
    bmi_range = f"{sub['BMI'].min():.1f}-{sub['BMI'].max():.1f}"
    age_range = f"{sub['Age'].min():.0f}-{sub['Age'].max():.0f}"

    # 生成描述：主要展示 BMI 和 Age 的范围
    desc_str = f"BMI:[{bmi_range}], 年龄:[{age_range}]"

    best_week, max_rate = analyze_risk_with_error(sub)

    if best_week != -1:
        advice = f"{best_week} 周"
        if max_rate < 0.9: advice += "*"  # 标记未达标
    else:
        advice = "样本不足"

    group_id = chr(65 + i)  # A, B, C...
    print(f"组 {group_id:<2} | {desc_str:<45} | {len(sub):<6} | {advice:<12}")

print("-" * 100)
print("注：建议周数带有 '*' 表示该组峰值达标率未到90%，建议采用该组的最优周数。")
print("请将上方表格数据填入论文表 5-1 和 表 5-2。")