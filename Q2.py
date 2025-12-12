import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

print("正在计算真实数据（修复单位版），请稍候...\n")

# 1. 读取数据
try:
    df = pd.read_excel('附件.xlsx', sheet_name=0)  # 读取男胎数据
except:
    print("❌ 错误：找不到文件 '附件.xlsx'！")
    exit()

# 2. 数据清洗
# 根据你的描述，V列是Y浓度。这里我们用列名索引更稳妥
# 假设 Excel 的列顺序没变，直接取第22列(索引21)作为Y浓度
try:
    col_map = {
        df.columns[9]: 'Week',  # J列
        df.columns[10]: 'BMI',  # K列
        df.columns[21]: 'Y_Conc'  # V列
    }
    df = df.rename(columns=col_map)
except:
    print("❌ 列索引错误，请检查Excel列数是否符合题目描述")
    exit()

# 提取数值
df['Week_Num'] = df['Week'].astype(str).str.extract(r'(\d+)').astype(float)
df['Y_Conc'] = pd.to_numeric(df['Y_Conc'], errors='coerce')
df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
df = df.dropna(subset=['Week_Num', 'BMI', 'Y_Conc'])

# 【关键修复步骤】：判断数据单位
# 如果最大值小于 1，说明是小数格式 (0.05)，需要乘以 100 变成百分数 (5)
if df['Y_Conc'].max() < 1:
    print("检测到数据为小数格式，正在转换为百分数...")
    df['Y_Conc'] = df['Y_Conc'] * 100

# 再次清洗异常值
df = df[df['Y_Conc'] > 0]

# 3. K-Means 聚类 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['BMI']])

# 按BMI均值排序：0=瘦, 1=中, 2=胖
cluster_map = df.groupby('Cluster')['BMI'].mean().sort_values().index
mapper = {old: new for new, old in enumerate(cluster_map)}
df['Group'] = df['Cluster'].map(mapper)
labels = ['偏瘦组 (低风险)', '正常组 (中风险)', '超重组 (高风险)']

# 计算是否达标 (现在单位统一了，可以直接比较)
df['Is_Qualified'] = (df['Y_Conc'] >= 4).astype(int)

# 4. 生成论文表格数据
print("=" * 20 + " 【表格数据：请更新论文 4.1 节表格】 " + "=" * 20)
print(f"| 组别 | BMI 聚类中心 | 划分区间 (kg/m^2) | 样本占比 |")
print(f"| :--- | :--- | :--- | :--- |")

group_stats = []
colors = ['#2ca02c', '#1f77b4', '#d62728']  # 绿、蓝、红

for i in range(3):
    sub = df[df['Group'] == i].copy()
    center = sub['BMI'].mean()
    min_b, max_b = sub['BMI'].min(), sub['BMI'].max()
    ratio = len(sub) / len(df) * 100
    print(f"| **{labels[i]}** | {center:.1f} | **{min_b:.1f} - {max_b:.1f}** | {ratio:.1f}% |")
    group_stats.append({'label': labels[i], 'data': sub, 'color': colors[i]})

print("\n" + "=" * 20 + " 【时点结论：请更新论文 4.2 节文字】 " + "=" * 20)

# 5. 计算最佳检测时点 & 画图
plt.figure(figsize=(10, 6))

for i in range(3):
    item = group_stats[i]
    # 按周统计达标率
    # 过滤掉样本太少的周(<10个)，防止波动太大
    stats = item['data'].groupby('Week_Num')['Is_Qualified'].agg(['mean', 'count'])
    stats = stats[stats['count'] >= 5]

    # 画图
    plt.plot(stats.index, stats['mean'], marker='o', label=item['label'], color=item['color'], linewidth=2)

    # 找最早达到 90% 的周
    qualified_weeks = stats[stats['mean'] >= 0.9]

    if not qualified_weeks.empty:
        best_week = int(qualified_weeks.index.min())
        curr_rate = qualified_weeks.loc[best_week, 'mean']
        print(f"✅ {item['label']}：建议在 **{best_week} 周** 检测 (达标率 {curr_rate:.1%})。")
    else:
        # 如果都没达到90%，找最高的
        if not stats.empty:
            best_week = int(stats['mean'].idxmax())
            max_rate = stats['mean'].max()
            print(
                f"⚠️ {item['label']}：未达到90%阈值，峰值在第 {best_week} 周 ({max_rate:.1%})。建议推迟至 **{best_week} 周**。")
        else:
            print(f"⚠️ {item['label']}：有效样本不足，无法计算。")

plt.axhline(y=0.9, color='gray', linestyle='--', label='90% 达标阈值')
plt.title('图4：各BMI组孕妇检测达标率随孕周变化趋势')
plt.xlabel('孕周 (Week)')
plt.ylabel('Y染色体浓度达标率 (Ratio >= 4%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)  # 固定Y轴范围0-1
plt.savefig('Figure4_Real_Data_Fixed.png')
print(f"\n[提示] 修复版趋势图已生成为 'Figure4_Real_Data_Fixed.png'。")
print("=" * 60)