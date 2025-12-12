import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel('附件.xlsx', sheet_name=0)
except Exception as e:
    exit()

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

if df['Week'].dtype == object:
    df['Week_Num'] = df['Week'].astype(str).str.extract(r'(\d+)').astype(float)
else:
    df['Week_Num'] = df['Week']

for c in ['Age', 'Height', 'Weight', 'BMI', 'Y_Conc']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df = df.dropna(subset=['Age', 'Height', 'Weight', 'BMI', 'Y_Conc', 'Week_Num'])

if df['Y_Conc'].median() < 1:
    df['Y_Conc'] = df['Y_Conc'] * 100
df = df[df['Y_Conc'] > 0]

feature_cols = ['BMI', 'Age', 'Height', 'Weight']
X = df[feature_cols]
y = df['Y_Conc']

tree_model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
tree_model.fit(X, y)
df['Leaf_Node'] = tree_model.apply(X)

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

leaf_means = df.groupby('Leaf_Node')['Y_Conc'].mean().sort_values(ascending=False)

for i, leaf in enumerate(leaf_means.index):
    sub = df[df['Leaf_Node'] == leaf]

    bmi_range = f"{sub['BMI'].min():.1f}-{sub['BMI'].max():.1f}"
    age_range = f"{sub['Age'].min():.0f}-{sub['Age'].max():.0f}"

    desc_str = f"BMI:[{bmi_range}], 年龄:[{age_range}]"

    best_week, max_rate = analyze_risk_with_error(sub)

    if best_week != -1:
        advice = f"{best_week} 周"

    group_id = chr(65 + i)
    print(f"组 {group_id:<2}  {desc_str:<45}  {len(sub):<6}  {advice:<12}")