import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

try:
    df = pd.read_excel('附件.xlsx', sheet_name=0)
except:
    exit()

try:
    col_map = {
        df.columns[9]: 'Week',
        df.columns[10]: 'BMI',
        df.columns[21]: 'Y_Conc'
    }
    df = df.rename(columns=col_map)
except:
    exit()

df['Week_Num'] = df['Week'].astype(str).str.extract(r'(\d+)').astype(float)
df['Y_Conc'] = pd.to_numeric(df['Y_Conc'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df = df.dropna(subset=['Week_Num', 'BMI', 'Y_Conc'])

if df['Y_Conc'].max() < 1:
    df['Y_Conc'] = df['Y_Conc'] * 100

df = df[df['Y_Conc'] > 0]

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['BMI']])

cluster_map = df.groupby('Cluster')['BMI'].mean().sort_values().index
mapper = {old: new for new, old in enumerate(cluster_map)}
df['Group'] = df['Cluster'].map(mapper)
labels = ['偏瘦组 (低风险)', '正常组 (中风险)', '超重组 (高风险)']

df['Is_Qualified'] = (df['Y_Conc'] >= 4).astype(int)

group_stats = []
colors = ['#2ca02c', '#1f77b4', '#d62728']
for i in range(3):
    sub = df[df['Group'] == i].copy()
    center = sub['BMI'].mean()
    min_b, max_b = sub['BMI'].min(), sub['BMI'].max()
    ratio = len(sub) / len(df) * 100
    print(f"{labels[i]} {center:.1f} {min_b:.1f} - {max_b:.1f} {ratio:.1f}%")
    group_stats.append({'label': labels[i], 'data': sub, 'color': colors[i]})

plt.figure(figsize=(10, 6))

for i in range(3):
    item = group_stats[i]
    stats = item['data'].groupby('Week_Num')['Is_Qualified'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= 5]

    plt.plot(stats.index, stats['mean'], marker='o', label=item['label'], color=item['color'], linewidth=2)

    qualified_weeks = stats[stats['mean'] >= 0.9]

    if not qualified_weeks.empty:
        best_week = int(qualified_weeks.index.min())
        curr_rate = qualified_weeks.loc[best_week, 'mean']
        print(f"{item['label']}：建议在 {best_week} 周检测 (达标率 {curr_rate:.1%})。")
    else:
        if not stats.empty:
            best_week = int(stats['mean'].idxmax())
            max_rate = stats['mean'].max()
            print(
                f"{item['label']}：未达到90%阈值，峰值在第 {best_week} 周 ({max_rate:.1%})。建议推迟至 {best_week}周。")
        else:
            print(f"{item['label']}：有效样本不足。")

plt.axhline(y=0.9, color='gray', linestyle='--', label='90% 达标阈值')
plt.title('图4：各BMI组孕妇检测达标率随孕周变化趋势')
plt.xlabel('孕周 (Week)')
plt.ylabel('Y染色体浓度达标率 (Ratio >= 4%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.savefig('Figure4_Real_Data_Fixed.png')