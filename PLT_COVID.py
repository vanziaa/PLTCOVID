# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import pymrmr
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import cross_val_score
from scipy.stats import mannwhitneyu
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
from statsmodels.formula.api import logit
from statsmodels.stats.mediation import Mediation
from scipy.stats import norm
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.ticker as ticker
import warnings
import time


warnings.filterwarnings("ignore")

# %%
def lassofunc(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    alphas = [0.01, 0.1,1,10,20]
    print("dataset splited")
    # 使用LassoCV进行交叉验证
    lasso_cv = LassoCV(alphas=alphas, cv=5)# 这里cv=5表示使用5折交叉验证
    print("lasso start")  
    #lasso_cv.fit(X_train, y_train)
    lasso_cv.fit(X, y)
    # 获取最佳的alpha值
    print("Best alpha:", lasso_cv.alpha_)
    # 打印系数
    print("Intercept:", lasso_cv.intercept_)
    print("Coefficients:", lasso_cv.coef_)
    # 使用最佳的alpha值在测试集上进行预测
    #y_pred = lasso_cv.predict(X_test)
    # 获取被保留的特征的索引
    selected_features_indices = np.where(lasso_cv.coef_ != 0)[0]
    print("Selected feature indices:", selected_features_indices)
    
    feature_names = X.columns.tolist()
    selected_feature_names = [feature_names[i] for i in selected_features_indices]
    print("Selected feature names:", selected_feature_names)
    # 过滤出系数不为0的特征和它们的系数
    print("image start")
    non_zero_coeffs = lasso_cv.coef_[lasso_cv.coef_ != 0]
    non_zero_feature_names = np.array(feature_names)[lasso_cv.coef_ != 0]
    palette = sns.dark_palette("brown", reverse=True)  # reverse=True 让它从浅到深
    # 创建条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x=non_zero_coeffs, y=non_zero_feature_names, palette=palette)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.title('Lasso Feature Selection (Non-Zero Coefficients Only)')
    #plt.axvline(x=0, color='red', linestyle='--')  # Add a vertical line at x=0 for reference
    plt.tight_layout()
    plt.show()
    print("finished")
    return selected_feature_names  # this is optiona
def ave_med(df):
    grouped = df.groupby('patient_id')
    average_df = grouped.mean().reset_index()
    average_df1 = pd.merge(average_df, df_patient, on='patient_id', how='left')
    median_df = grouped.median().reset_index()
    median_df1 = pd.merge(median_df, df_patient, on='patient_id', how='left')
    summary_df = pd.merge(average_df, median_df, on='patient_id', suffixes=('_mean', '_median'))
    summary_df = pd.merge(summary_df, df_patient, on='patient_id', how='left')
    return average_df1, median_df1, summary_df
def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)
def kurtosis(x):
    return x.kurtosis()
def stat_me(df,start,end):
    statistics = ['mean', 'std', 'median', 'skew']
    # 对除了前两列之外的所有列进行统计计算
    stats_df = df.groupby('patient_id')[df.columns[start:end]].agg(['mean', 'std', 'median', iqr, 'skew']).reset_index()
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df.rename(columns={'patient_id_':'patient_id'}, inplace = True)
    # 应用峰度计算
    kurtosis_df = df.groupby('patient_id')[df.columns[start:end]].apply(lambda x: x.kurt()).reset_index()
    kurtosis_df.columns = ['patient_id'] + [col + '_kurt' for col in df.columns[start:end]]
    stats_df = pd.merge(stats_df, kurtosis_df, on='patient_id')
    # 简化列名称
    
    # 保存到新的 CSV 文件
    stats_df.to_csv('patient_statistics.csv', index=False)
    return stats_df
def plt_kde(selected_feature_names,a,b,df,valename):
    fig, axs = plt.subplots(a, b, figsize=(15, 15), sharey=False) 
    colors = ['red', 'blue', 'green']  # 为每组数据分配颜色
    for idx, feature in enumerate(selected_feature_names):
        ax = axs[idx // a, idx % b]
        for group, color in zip([0, 1, 2], colors):
            data_group = df[df[valename] == group][feature].dropna() 
            sns.kdeplot(data_group, ax=ax, color=color, label=f'Group {group}', fill=True)
        lower_bound = np.percentile(df[feature].dropna(), 2.5)
        upper_bound = np.percentile(df[feature].dropna(), 97.5)
        ax.set_xlim(lower_bound, upper_bound)
        ax.set_title(feature.replace("_DF_image", ""))
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    plt.tight_layout()
    plt.show()
def mrmrfuc (X,y,n):
    dfmrmr = y.join(X)
    dfmrmr = dfmrmr.dropna()
    #print("selecting features...")
    selected_feature_names  = pymrmr.mRMR(dfmrmr, 'MIQ', n)
    #print(selected_feature_names)
    return selected_feature_names 
def NORM (df):
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)
    df = pd.DataFrame(df_normalized, columns=df.columns)
    return df   

# %%
# 读取 CSV 文件
df_p1 = pd.read_csv(r'I:\qianyu\result\MyExpt_comi_DF_1.csv')
df_p2 = pd.read_csv(r'I:\qianyu\result\MyExpt1107_comii_DF_1.csv')
df_p3 = pd.read_csv(r'I:\qianyu\result\MyExpt1107_comiii_DF_1.csv')
df_p4 = pd.read_csv(r'I:\qianyu\result\MyExpt1115_comiiii_DF_1.csv')
df_p5 = pd.read_csv(r'I:\qianyu\result\MyExpt1124_comiiii_DF_1.csv')
# 比较df1和df2的列名和顺序
columns_identical = list(df_p1.columns) == list(df_p5.columns)
# 输出结果
print("df1和df5的列名和顺序是否完全一致:", columns_identical)
df1 = pd.concat([df_p1, df_p2, df_p3, df_p4, df_p5], ignore_index=True)
#df2 = pd.concat([df_p1, df_p2, df_p3], ignore_index=True)
df1['patient_id'] = df1['FileName_DF_image'].str.extract(r'^(\d+)')
#df2['patient_id'] = df2['FileName_DF_image'].str.extract(r'^(\d+)')
first_col = df1.pop('patient_id')
df1.insert(0, 'patient_id', first_col)
#first_col = df2.pop('patient_id')
#df2.insert(0, 'patient_id', first_col)
df1 = df1.drop(df1.loc[:, 'ImageNumber':'PathName_DF_image'].columns, axis=1)
df1 = df1.drop(df1.loc[:, 'AreaShape_BoundingBoxArea':'AreaShape_Center_Y'].columns, axis=1)
df1 = df1.drop(df1.loc[:, 'Location_CenterMassIntensity_X_DF_image':'Number_Object_Number'].columns, axis=1)

# %%
# 计算每个病人中面积大于72的血小板的百分比
grouped = df1.groupby('patient_id')

# 计算大于72的血小板数量和总数量
platelet_count_greater_than_75 = grouped['AreaShape_Area'].apply(lambda x: (x > 75).sum())
total_platelet_count = grouped['AreaShape_Area'].count()

# 计算百分比
percentage = (platelet_count_greater_than_75 / total_platelet_count) * 100
percentage.name = 'percentage_greater_than_75'

# 将结果整合到一个DataFrame中
resultPAR = pd.DataFrame(percentage).reset_index()

# %%
del df_p1, df_p2, df_p3, df_p4, first_col

# %%
#df_patient = pd.read_excel('I:\\qianyu\\data_ana_613.xlsx',sheet_name='Sheet1')
df_patient = pd.read_excel('I:\\qianyu\\data_ana_1125.xlsx',sheet_name='Sheet1')
#df_patient = pd.read_excel('I:\\qianyu\\data_ana_1117.xlsx',sheet_name='Sheet5')
df_patient['patient_id'] = df_patient['patient_id'].astype(str)
print("Number of rows:", df_patient.shape[0],"Number of columns:", df_patient.shape[1])

# %%
df1_stat_ori = stat_me(df1,1,df1.shape[1])

# %%
# 删除这些列
df1.drop(columns='AreaShape_FormFactor', inplace=True)


# %%
df1

# %%
import pandas as pd
import numpy as np
from scipy.stats import iqr, kurtosis, skew, entropy

def calculate_histogram_features(x):
    if len(x.dropna()) == 0:
        return [np.nan] * 16
    energy = np.sum(np.square(x))
    total_energy = energy / len(x)
    entropy_val = entropy(x.value_counts(normalize=True), base=2) if len(np.unique(x)) > 1 else np.nan
    percentile_10th = np.percentile(x, 10)
    percentile_90th = np.percentile(x, 90)
    mean = np.mean(x)
    median = np.median(x)
    interquartile_range = iqr(x)
    mean_absolute_deviation = np.mean(np.abs(x - mean))
    robust_mad = np.median(np.abs(x - np.median(x))) / 0.6745
    root_mean_squared = np.sqrt(np.mean(np.square(x)))
    skewness = skew(x, nan_policy='omit')
    standard_deviation = np.std(x)
    kurtosis_val = kurtosis(x, nan_policy='omit')
    variance = np.var(x)
    uniformity = 1 - len(np.unique(x)) / len(x)
    
    return [energy, total_energy, entropy_val, percentile_10th, percentile_90th, mean, median, interquartile_range, mean_absolute_deviation, robust_mad, root_mean_squared, skewness, standard_deviation, kurtosis_val, variance, uniformity]

def stat_me(df, start, end):
    features_names = ['Energy', 'Total Energy', 'Entropy', '10th Percentile', '90th Percentile', 'Mean', 'Median', 'Interquartile Range', 'Mean Absolute Deviation', 'Robust MAD', 'Root Mean Squared', 'Skewness', 'Standard Deviation', 'Kurtosis', 'Variance', 'Uniformity']
    patient_rows = []  # 用于收集每个病人的特征数据

    columns = df.columns[start:end]
    
    for patient_id, group in df.groupby('patient_id'):
        patient_data = {'patient_id': patient_id}
        for column in columns:
            hist_features = calculate_histogram_features(group[column].dropna())
            for name, value in zip(features_names, hist_features):
                patient_data[f"{column}_{name}"] = value
        patient_rows.append(patient_data)
    
    result_df = pd.DataFrame(patient_rows)
    result_df.set_index('patient_id', inplace=True)
    
    result_df.to_csv('patient_histogram_features2.csv')
    
    return result_df

# %%
df1_stat_ori = stat_me(df1,3,df1.shape[1])

# %%
df1_stat_ori

# %%
merged_df = pd.merge(df1_stat_ori, df_patient[['patient_id', 'covid2', 'mbs4c1']], on='patient_id', how='left')


# %%
merged_df

# %%
# 获取所有列名的列表
columns = list(merged_df.columns)

# 重新排列列的顺序：先是第一列，然后是最后两列，最后是第二列到倒数第三列
new_order = [columns[0]] + columns[-2:] + columns[1:-2]

# 使用新的列顺序重新索引DataFrame
df1 = merged_df[new_order]

# %%
df1

# %%
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


features = df1.iloc[:, 3:]
#features = features.dropna()


# %%
features['label'] = (features['AreaShape_Area'] >= 48).astype(int)

# 如果你想将这个新列 "label" 移动到 DataFrame 的最前面
cols = features.columns.tolist()
cols = cols[-1:] + cols[:-1]  # 将最后一个列名移动到列表的开始位置
features = features[cols]

# %%
# 找出含有无穷大值的列
columns_with_inf = [col for col in features.columns if np.isinf(features[col]).any()]

# 删除这些列
features.drop(columns=columns_with_inf, inplace=True)

# 输出被删除的列名
print("删除的列名:", columns_with_inf)

# %%
total_rows = features.shape[0]  # 数据的总行数
desired_sample_size = total_rows // 100  # 计算总样本数的1%

# 计算抽样间隔
interval = total_rows // desired_sample_size

# 使用间隔进行抽样
sampled_data = features.iloc[::interval]

# 根据需要调整抽样大小，确保最终样本大小接近目标
# 注意：根据间隔抽样可能会导致实际抽样数略有偏差
sampled_features = sampled_data.iloc[:desired_sample_size]

# %%

sampled_features_covid=sampled_features['covid2'].astype(int)
sampled_features_mbs=sampled_features['mbs4c1'].astype(int)
sampled_features=sampled_features.iloc[:, 3:]

# %%
print(sampled_features_covid)

# %%
sampled_features = NORM(sampled_features)


# %%
from sklearn.decomposition import PCA
n_components = 50
pca = PCA(n_components=n_components,svd_solver='randomized', random_state=42)
reduced_data = pca.fit_transform(sampled_features)
# 使用t-SNE进行降维
tsne = TSNE(n_components=3, random_state=42)
features_3d = tsne.fit_transform(reduced_data)


# %%
tsne = TSNE(n_components=2, random_state=42,perplexity=50)
#features_2d = tsne.fit_transform(reduced_data)
features_2d = tsne.fit_transform(sampled_features)

# %%
unique_labels

# %%
df1_stat = pd.merge(df1_stat_ori, df_patient, on='patient_id', how='left')
df1_stat = pd.merge(resultPAR, df1_stat, on='patient_id', how='left')
print("Size of df1_stat:", df1_stat.shape)
start_col_index = df1_stat.columns.get_loc('patient_id')
end_col_index = df1_stat.columns.get_loc('Texture_Variance_DF_image_3_03_256_Uniformity')
# 使用 .loc 从起始列到结束列提取所有列
X = df1_stat.iloc[:, start_col_index:end_col_index + 1] 
X['patient_id'].astype(int)
print("Size of X :", X.shape)
y1 = df1_stat[['mbs4c1']]
y2 = df1_stat[['covid2']]
# 找出包含缺失值的列
na_columns = X.columns[X.isna().any()].tolist()
print("na columns",na_columns)
X.drop(columns=na_columns, inplace=True)
print("na columns dropped")
print("Size of X :", X.shape)
print("Size of y1 :", y1.shape)
print("Size of y2 :", y2.shape)
df_in = y1.join(y2)
df_in = df_in.join(X)
print("Size of df_in :", df_in.shape)
df_in = df_in.dropna()
df_in.reset_index(drop=True, inplace=True)
print("Size of df_in :", df_in.shape)

# %%
df_in

# %%
start_col_index = df_in.columns.get_loc('percentage_greater_than_75')
end_col_index = df_in.columns.get_loc('Texture_Variance_DF_image_3_03_256_Uniformity')
pid = df_in[["patient_id"]]
X = df_in.iloc[:, start_col_index:end_col_index]  # 所有行，除了最后两列
print("Size of X :", X.shape)
y_mb = df_in[['mbs4c1']]
print("Size of y_mb :", y_mb.shape)
y_covid = df_in[['covid2']]
print("Size of y_covid :", y_covid.shape)
x1 = X
X = NORM(X)

# %%
# 提取并映射新旧列名
new_column_names = {}
new_columns = []
old_columns = X.columns.tolist()
for col in old_columns:
    feature_category = col.split('_')[0]
    if feature_category not in new_column_names:
        new_column_names[feature_category] = 1
    else:
        new_column_names[feature_category] += 1
    new_columns.append(f"{feature_category}_{new_column_names[feature_category]}")

# 创建新的DataFrame以存储新旧列名对应关系
column_mapping_df = pd.DataFrame({
    'Old Column Name': old_columns,
    'New Column Name': new_columns
})

# 重命名原始DataFrame的列
X.columns = new_columns

#print("Column Mapping:")
#print(column_mapping_df)
print("\nUpdated DataFrame:")
#print(X)

# %%
coefficients = lasso_model.coef_
intercept = lasso_model.intercept_
combined_features_tr = pd.concat((X_tr[mr_shape_covid0],X_tr[mr_Intensity_covid0], X_tr[mr_text_covid0]), axis=1)
linear_combination_tr = np.dot(combined_features_tr, 
                            coefficients.T) + intercept
combined_features_ts = pd.concat((X_ts[mr_shape_covid0],X_ts[mr_Intensity_covid0], X_ts[mr_text_covid0]), axis=1)
linear_combination_ts = np.dot(combined_features_ts, 
                            coefficients.T) + intercept

# %%
from statistics import median


def calculate_best_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    best_threshold = thresholds[youden_index.argmax()]
    best_threshold = median(thresholds)
    return best_threshold

def calculate_sensitivity_specificity(y_true, y_pred):
    true_positive = ((y_pred == 1) & (y_true == 1)).sum()
    true_negative = ((y_pred == 0) & (y_true == 0)).sum()
    false_positive = ((y_pred == 1) & (y_true == 0)).sum()
    false_negative = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return sensitivity, specificity

# 假设你已经有了这些变量：y_train, y_test, y_prob_tr, y_prob_ts

# 计算最佳阈值
best_threshold = calculate_best_threshold(y_tr, linear_combination_tr)
print(best_threshold)
# 应用阈值进行分类
y_pred_tr = (linear_combination_tr >= best_threshold).astype(int)
y_pred_ts = (linear_combination_ts >= best_threshold).astype(int)

# 计算敏感性和特异性
sensitivity_tr, specificity_tr = calculate_sensitivity_specificity(y_tr, y_pred_tr)
sensitivity_ts, specificity_ts = calculate_sensitivity_specificity(y_ts, y_pred_ts)

print(f"Training Sensitivity: {sensitivity_tr}, Training Specificity: {specificity_tr}")
print(f"Test Sensitivity: {sensitivity_ts}, Test Specificity: {specificity_ts}")

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# %%
plt.rcParams.update({'font.size': 10}) 
plt.rcParams['font.family'] = 'Calibri'

# 假设您已经有了一个名为 X 的数据集并已经进行了适当的预处理。

# 应用 K-means 聚类
kmeans = KMeans(n_clusters=2,random_state=1)
kmeans.fit(X)
labels = kmeans.labels_

# 应用 PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 分离出不同的簇
cluster_1 = X_pca[labels == 0]
cluster_2 = X_pca[labels == 1]

# 创建绘图
fig, ax = plt.subplots(figsize=(4, 3))

# 绘制点
ax.scatter(cluster_1[:, 0], cluster_1[:, 1], c='#FA9300', linewidths=1, edgecolor='black', label='Cluster 1')
ax.scatter(cluster_2[:, 0], cluster_2[:, 1], c='#0148FA', linewidths=1, edgecolor='black', label='Cluster 2')

# 可选：绘制簇的多边形
def draw_polygon_around_points(ax, points, **kwargs):
    hull = ConvexHull(points)
    poly = Polygon(points[hull.vertices], **kwargs)
    ax.add_patch(poly)

# 画出围绕簇的多边形
draw_polygon_around_points(ax, cluster_1, facecolor='#FA9300', alpha=0.2)
draw_polygon_around_points(ax, cluster_2, facecolor='#0148FA', alpha=0.2)

# 设置图例
#ax.legend()
ax.tick_params(axis='x', which='major', pad=-4,labelsize=10)
ax.tick_params(axis='y', which='major', pad=-4,labelsize=10)
# 设置坐标轴标题
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
plt.tight_layout()
fig.savefig("129_kmeans_scatter.svg", format='svg')
# 显示图像
plt.show()

# %%
from sklearn.metrics import pairwise_distances
import numpy as np

def average_nearest_neighbor_distance(X, labels):
    distances = pairwise_distances(X)
    avg_distance = []
    for i in range(len(X)):
        same_cluster_indices = labels == labels[i]
        same_cluster_indices[i] = False  # exclude the point itself
        avg_distance.append(np.mean(distances[i, same_cluster_indices]))
    return np.mean(avg_distance)

ann_distance = average_nearest_neighbor_distance(X, kmeans.labels_)

# %%
kmeans.labels_

# %%
from scipy.stats import chi2_contingency
plt.rcParams.update({'font.size': 10}) 
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['savefig.dpi'] = 900
df = df_an2
df['cluster2'] = kmeans.labels_
df['cluster2'] = 'Cluster ' + df['cluster2'].astype(str)
# 计算每个分类变量中每个聚类的百分比
percentage_category1 = df.groupby('mbs4c1')['cluster2'].value_counts(normalize=True).unstack().fillna(0) * 100
percentage_category2 = df.groupby('covid2')['cluster2'].value_counts(normalize=True).unstack().fillna(0) * 100

# 将分类变量转换为字符串
#df['mbs4c1'] = df['mbs4c1'].map({0: 'a', 1: 'b', 2: 'c'})
#df['covid2'] = df['covid2'].map({0: 'a', 1: 'b'})

# 定义堆叠柱状图绘制函数
def plot_stacked_bar(percentage, category_labels, ax):
    bar_width = 0.4  # 柱状图的宽度
    index = np.arange(len(category_labels))
    
    # 绘制每个聚类的堆叠柱状图
    ax.bar(index, percentage['Cluster 0'], bar_width, 
           color='#FA9300', 
           label='Cluster 0')
    ax.bar(index, percentage['Cluster 1'], bar_width, 
           bottom=percentage['Cluster 0'], color='#0148FA', 
           label='Cluster 1')

    ax.set_xticks(index)
    ax.set_xticklabels(category_labels)
    ax.set_ylabel('Percentage (%)')

# 创建一个 2 行 3 列的图表布局
fig, axs = plt.subplots(2, 3, figsize=(4, 4))


# 第一行的第三个堆叠柱状图
plot_stacked_bar(percentage_category1, ['a', 'b', 'c'], axs[0, 2])

# 第二行的第三个堆叠柱状图
plot_stacked_bar(percentage_category2, ['a', 'b'], axs[1, 2])
df['cluster'] = df['cluster2'].map({'Cluster 0': 1, 'Cluster 1': 2})
# 第一行的前两个箱型图
sns.boxplot(x='cluster', y='plt', data=df, 
            palette=["#FA9300", "#0148FA"], showfliers=False, 
            ax=axs[0, 0], width=0.5)
sns.boxplot(x='cluster', y='aptt', data=df, 
            palette=["#FA9300", "#0148FA"], showfliers=False, 
            ax=axs[0, 1], width=0.5)

# 第二行的前两个箱型图
sns.boxplot(x='cluster', y='inr', data=df, 
            palette=["#FA9300", "#0148FA"], showfliers=False, 
            ax=axs[1, 0], width=0.5)
sns.boxplot(x='cluster', y='plt_dd', data=df, 
            palette=["#FA9300", "#0148FA"], showfliers=False, 
            ax=axs[1, 1], width=0.5)
# 隐藏所有子图的图例
for ax in axs.flat:
    ax.legend().set_visible(False)

# 设置所有子图的字体大小和轴标签
for ax in axs.flat:
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='major', pad=-4,labelsize=10)
    ax.tick_params(axis='y', which='major', pad=-4)

# 调整布局
plt.tight_layout()
fig.savefig("129_kmeans_compare.svg", format='svg')
# 显示图形
plt.show()

# %%


# %%
from sklearn.metrics import roc_curve, auc
y_prob_ts = lasso_model.predict_proba(combined_features_ts)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_ts, y_prob_ts)
y_prob_tr = lasso_model.predict_proba(combined_features_tr)[:, 1]
fprtr, tprtr, thresholdstr = roc_curve(y_tr, y_prob_tr)
plt.rcParams['font.family'] = 'Calibri'
# 计算 AUC
roc_auc = auc(fpr, tpr)
roc_auc_tr = auc(fprtr, tprtr)
# 绘制 ROC 曲线
plt.figure(figsize=(8/2.54, 5.5/2.54))
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Calibri'
sns.set(style="whitegrid", rc={'axes.edgecolor': 'black', 'axes.linewidth': 0.75})
plt.plot(fprtr, tprtr, color='#164989', lw=2, label='Training (AUC = %0.2f)' % roc_auc_tr)
plt.plot(fpr, tpr, color='red', lw=2, label='Test (AUC = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-specificity')
plt.ylabel('Sensitivity')
ax = plt.gca()
ax.tick_params(axis='x', which='major', pad=-1)  # X轴
ax.tick_params(axis='y', which='major', pad=-1) 
#plt.title('Receiver Operating Characteristic Example')
plt.legend(loc='lower right', fontsize=10)
plt.savefig("129_ROC.svg", format='svg')
plt.show()

# %%
# 获取系数
coefficients = lasso_model.coef_[0]  # 对于逻辑回归，coef_ 是一个二维数组
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams.update({'font.size': 10})
# 如果你有一个特征名列表可以这样获取
feature_names = combined_features_tr.columns.tolist()

# 选择系数不为零的特征和系数
non_zero_indices = [i for i, coef in enumerate(coefficients) if coef != 0]
non_zero_features = [feature_names[i] for i in non_zero_indices]
non_zero_coefficients = [coefficients[i] for i in non_zero_indices]

# 组合不为零的系数和对应的特征
non_zero_coefficients_features = dict(zip(non_zero_features, non_zero_coefficients))

# 打印不为零的系数和对应的特征
for feature, coef in non_zero_coefficients_features.items():
    print(f'{feature}: {coef}')
colors = ['#164989' if coef < 0 else '#6b302f' for coef in non_zero_coefficients]
plt.figure(figsize=(5/2.54, 6/2.54))
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Calibri'
sns.set(style="whitegrid", rc={'axes.edgecolor': 'black', 'axes.linewidth': 0.75})
#non_zero_features = [label.replace("_", "\n") for label in non_zero_features]
sns.barplot(x=non_zero_coefficients, y=non_zero_features, palette=colors)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
ax = plt.gca()
ax.tick_params(axis='x', which='major', pad=-4)  # X轴
ax.tick_params(axis='y', which='major', pad=-4)  # Y轴，如果需要的话
#plt.title('Lasso Feature Selection (Non-Zero Coefficients Only)')
    #plt.axvline(x=0, color='red', linestyle='--')  # Add a vertical line at x=0 for reference
#plt.tight_layout()
plt.savefig("129_featurename.svg", format='svg')
plt.show()
print("finished")

# %%
coefficients = lasso_model.coef_
intercept = lasso_model.intercept_
combined_features = pd.concat((X[mr_shape_covid0],X[mr_Intensity_covid0], X[mr_text_covid0]), axis=1)
linear_combination = np.dot(combined_features, 
                            coefficients.T) + intercept
#linear_combination = lasso_model.predict_proba(combined_features)[:, 1]
y_probdf = pd.DataFrame(linear_combination, columns=['PLT_score'])
y_probdf = y_probdf.join(pid)
y_probdf = y_probdf.join(y_mb)
df_an = y_probdf.join(y_covid)
df_an.to_csv('AN_TEST22.csv', index=False)

# %%
combined_features

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(combined_features)
# 分离出不同的簇
cluster_1 = X_pca[labels == 0]
cluster_2 = X_pca[labels == 1]

# 创建绘图
fig, ax = plt.subplots(figsize=(4, 3))

# 绘制点
ax.scatter(cluster_1[:, 0], cluster_1[:, 1], c='#FA9300', linewidths=1, edgecolor='black', label='Cluster 1')
ax.scatter(cluster_2[:, 0], cluster_2[:, 1], c='#0148FA', linewidths=1, edgecolor='black', label='Cluster 2')

# 可选：绘制簇的多边形
def draw_polygon_around_points(ax, points, **kwargs):
    hull = ConvexHull(points)
    poly = Polygon(points[hull.vertices], **kwargs)
    ax.add_patch(poly)

# 画出围绕簇的多边形
draw_polygon_around_points(ax, cluster_1, facecolor='#FA9300', alpha=0.2)
draw_polygon_around_points(ax, cluster_2, facecolor='#0148FA', alpha=0.2)

# 设置图例
#ax.legend()
ax.tick_params(axis='x', which='major', pad=-4,labelsize=10)
ax.tick_params(axis='y', which='major', pad=-4,labelsize=10)
# 设置坐标轴标题
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
plt.tight_layout()
fig.savefig("129_kmeans_scatter.svg", format='svg')
# 显示图像
plt.show()

# %%
df_patient = pd.read_excel('I:\\qianyu\\data_ana_1125.xlsx',sheet_name='Sheet1')
#df_patient = pd.read_excel('I:\\qianyu\\data_ana_1117.xlsx',sheet_name='Sheet5')
df_patient['patient_id'] = df_patient['patient_id'].astype(str)
#print("Number of rows:", df_patient.shape[0],"Number of columns:", df_patient.shape[1])


# %%
df_heatmap

# %%
df_an['cluster'] = kmeans.labels_

# %%
sns.set(style="whitegrid", rc={'axes.edgecolor': 'black', 'axes.linewidth': 0.75})
 # Darker blue for 1, lighter blue for 0

plt.rcParams['font.family'] = 'Calibri'
# Create all pairs for the three categories
category_pairs = list(itertools.combinations([0, 1, 2], 2))

# Perform Mann-Whitney U test for each pair
mw_results = {}
for pair in category_pairs:
    category1 = pair[0]
    category2 = pair[1]
    test_stat, p_value = mannwhitneyu(df_an[df_an['mbs4c1'] == category1]['PLT_score'], 
                                      df_an[df_an['mbs4c1'] == category2]['PLT_score'])
    mw_results[f'Category {category1} vs Category {category2}'] = p_value

mw_results_corrected = {key: value * len(category_pairs) for key, value in mw_results.items()} # Bonferroni correction
mw_results_corrected

palette = sns.dark_palette("#79C", reverse=True)
# Function to add significance markers on the plot
def add_significance_marker(ax, x1, x2, y, h, text):
    """Adds a significance marker between two points in the plot."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
    ax.text((x1+x2)*.5, y+h-0.3, text, ha='center', va='bottom', color='black', fontsize=12)

# Plotting the boxplot with significance markers
plt.figure(figsize=(3, 3))
sns.violinplot(x='mbs4c1', y='PLT_score', data=df_an, palette=palette, width=0.3)
#plt.title('Violin Plot of PLT_score for Each Metabolic Category')
plt.xlabel('Number of metabolic disorders',fontsize=10)
plt.ylabel('Plt Score',fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xticks([0, 1, 2], ['0', '1~2', '≥3'])
ax = plt.gca()
ax.tick_params(axis='x', which='major', pad=-4)  # X轴
ax.tick_params(axis='y', which='major', pad=-4)
plt.grid(True)
h_factor =0.1
# Adding significance markers based on the corrected p-values
max_y = df_an['PLT_score'].max() + h_factor
min_y = df_an['PLT_score'].min() - h_factor
h = (max_y - min_y) * h_factor
for pair, p_value in mw_results_corrected.items():
    x1, x2 = [int(x[-1]) for x in pair.split(' vs ')]
    y = max_y + h * (x2 - x1 - 0.5)
    text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    add_significance_marker(plt.gca(), x1, x2, y, h, text)
plt.tight_layout()
plt.savefig("102_metabolic_compare.svg", format='svg')
plt.show()

# %%
import pandas as pd

# 假设 df_an 是你的数据框架
# 对 mbs4c1 的每个类别计算 PLT_score 的中位数和 IQR

# 分组并计算中位数
stats_df = df_an.groupby('mbs4c1')['PLT_score'].agg(median='median')

# 计算第25和第75百分位数
stats_df['q25'] = df_an.groupby('mbs4c1')['PLT_score'].quantile(0.25)
stats_df['q75'] = df_an.groupby('mbs4c1')['PLT_score'].quantile(0.75)

# 计算 IQR
stats_df['IQR'] = stats_df['q75'] - stats_df['q25']

# 删除不再需要的 q25 和 q75 列
stats_df = stats_df.drop(['q25', 'q75'], axis=1)

# 查看结果
print(stats_df)


# %%
X_sorted

# %%
# Set style
sns.set(style="whitegrid", rc={'axes.edgecolor': 'black', 'axes.linewidth': 0.75})
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Calibri'
# Let's assume df_an is a predefined DataFrame with your data
# Replace 'group1' and 'group2' with your actual group column names or conditions
group1_data = df_an[df_an['covid2'] == 0]['PLT_score']
group2_data = df_an[df_an['covid2'] == 1]['PLT_score']

# Perform Mann-Whitney U test for the two groups
test_stat, p_value = mannwhitneyu(group1_data, group2_data)

# Apply Bonferroni correction if necessary (remove if not needed)
p_value_corrected = p_value # Since there are two comparisons

# Set the palette for plotting
palette = sns.dark_palette("#79C", reverse=True)

# Function to add significance markers on the plot
def add_significance_marker(ax, x1, x2, y, h, text):
    """Adds a significance marker between two points in the plot."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c='black')
    ax.text((x1+x2)*.5, y+h, text, ha='center', va='bottom', color='black', fontsize=12)

# Plotting the violin plot with significance markers
plt.figure(figsize=(3, 3))
sns.violinplot(x='covid2', y='PLT_score', data=df_an[df_an['covid2'].isin([0, 1])], palette=palette, width=0.3)
plt.xlabel('', fontsize=10)
plt.ylabel('PLT Score', fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.xticks([0, 1], ['Non-severe\nCOVID-19', 'Severe\nCOVID-19'])
ax = plt.gca()
ax.tick_params(axis='x', which='major', pad=-4)
ax.tick_params(axis='y', which='major', pad=-4)
ax.set_ylim([-5, 6])
plt.grid(True)

## Adding significance marker based on the corrected p-value
max_y = max(group1_data.max(), group2_data.max())
min_y = min(group1_data.min(), group2_data.min())
h = (max_y - min_y) * 0.1
y = max_y + h+1
text = '***' if p_value_corrected < 0.001 else '**' if p_value_corrected < 0.01 else '*' if p_value_corrected < 0.05 else 'ns'
add_significance_marker(ax, 0, 1, y, h, text)
plt.tight_layout()
plt.savefig("102_COVID_compare.svg", format='svg')
plt.show()

# %%
for pair, p_value in mw_results_corrected.items():
    if p_value<0.05:
        print(p_value)

# %%

# Sorting the data by PLT_score
sorted_data = df_an.sort_values(by='PLT_score').reset_index(drop=True)
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Calibri'
plt.figure(figsize=(8/2.54, 5/2.54))
sns.set(style="whitegrid", rc={'axes.edgecolor': 'black', 'axes.linewidth': 0.75})
# Preparing colors based on covid2 values
colors = ['#164989' if x == 0 else 'red' for x in sorted_data['covid2']]

# Plotting with adjusted settings

sns.barplot(x=sorted_data.index, 
            y=sorted_data['PLT_score'], 
            palette=colors, 
            linewidth=0, edgecolor='none',width=1.0)
plt.xlabel('')
plt.ylabel('Plt Score')
plt.xticks([])
#plt.title('PLT Scores Sorted from Lowest to Highest (Colored by Covid2 Status)')

unique_mbs4c1 = sorted_data['mbs4c1'].unique()
mbs4c1_palette = sns.color_palette("hsv", len(unique_mbs4c1))
mbs4c1_color_lookup = {value: mbs4c1_palette[i] for i, value in enumerate(unique_mbs4c1)}
mbs4c1_colorbar = [mbs4c1_color_lookup[value] for value in sorted_data['mbs4c1']]


ax = plt.gca()
#ax.tick_params(axis='x', which='major', pad=-4)  # X轴
ax.tick_params(axis='y', which='major', pad=-4)  # Y轴，如果需要的话
#plt.title('Lasso Feature Selection (Non-Zero Coefficients Only)')
# Show the plot


plt.show()

# %%
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 10}) 
plt.rcParams['font.family'] = 'Calibri'

data = df_an
# Sorting the data by PLT_score
sorted_data = data.sort_values(by='PLT_score').reset_index(drop=True)

# Preparing colors based on covid2 values
covid_colors = ['#164989' if x == 0 else '#c1282d' for x in sorted_data['covid2']]

# Preparing colors based on mbs4c1 values for the colorbar
unique_mbs4c1 = sorted_data['mbs4c1'].unique()
mbs4c1_color_lookup = {
    0.0: '#d2e4ef',  # 指定的颜色可以是颜色名字
    1.0: '#4d93c3',  # 也可以是十六进制代码
    2.0: '#103d5a'
}

# 将颜色名称转换为RGBA格式
mbs4c1_color_lookup = {key: mcolors.to_rgba(color) for key, color in mbs4c1_color_lookup.items()}
mbs4c1_colorbar = [mbs4c1_color_lookup[value] for value in sorted_data['mbs4c1']]




# Plotting the sorted PLT_scores with a colorbar
fig, ax = plt.subplots(figsize=(20/2.54, 5/2.54))

# Create the bar plot for PLT_scores
sns.barplot(x=sorted_data.index, 
            y=sorted_data['PLT_score'],
              palette=covid_colors, width=1.0,
            linewidth=0, dodge=False, ax=ax)
from matplotlib.patches import Patch
# Create custom legends
covid_legend_elements = [Patch(facecolor='#164989', label='Covid2: 0'),
                         Patch(facecolor='#c1282d', label='Covid2: 1')]
mbs4c1_legend_elements = [Patch(facecolor=mbs4c1_color_lookup[value], label=f'mbs4c1: {value}') 
                          for value in unique_mbs4c1]

# Place the legend on the plot
#legend1 = ax.legend(handles=covid_legend_elements, title='Covid2 Status', loc='upper left')
#legend2 = ax.legend(handles=mbs4c1_legend_elements, title='mbs4c1 Values', loc='center', bbox_to_anchor=(0.6, -0.5))
#ax.add_artist(legend1)  # add the first legend back after it gets replaced by the second

#ax.legend(handles=legend_elements, title='Status')
ax.set_xlabel('')
ax.set_ylabel('Plt Score')
ax.set_xticks([])
#ax.set_title('PLT Scores Sorted from Lowest to Highest (Colored by Covid2 Status)')
ax.tick_params(labelsize=10)  # Set the font size of the ticks
ax.tick_params(axis='y', which='major', pad=-1)
# Positioning the colorbar axis below the main plot
barplot_pos = ax.get_position()
colorbar_height = 0.1  # Height of the colorbar
colorbar_pos = [barplot_pos.x0, barplot_pos.y0 - colorbar_height - 0.08, 
                barplot_pos.width, colorbar_height]

# Add the colorbar below the main plot
colorbar_ax = fig.add_axes(colorbar_pos)
colorbar_ax.imshow([mbs4c1_colorbar], aspect='auto')
colorbar_ax.set_axis_off()
fig.savefig("129_classfy.svg", format='svg')
plt.show()


# %%
mbs4c1_color_lookup

# %%
mbs4c1_colorbar

# %%
df_covariate = df1_stat[['age', 'sex', 'cld', 'plt', 'aptt', 'inr', 'plt_dd'
                         , 'ALT', 'CRP', 'WBC', 'Hb','treat']]

# %%
for column in df_covariate.columns:
    # 检测列的数据类型或唯一值的数量来判断是否为分类变量
    if df_covariate[column].dtype == 'object' or len(df_covariate[column].unique()) < 10: 
        # 分类变量：使用众数进行插补
        mode = df_covariate[column].mode()[0]
        df_covariate[column].fillna(mode, inplace=True)
    else:
        # 连续变量：使用中位数进行插补
        median = df_covariate[column].median()
        df_covariate[column].fillna(median, inplace=True)


# %%
df_covariate

# %%
df_an2 = df_an.join(df_covariate)
#df_an2 = df_an2.join(combined_features[non_zero_features])
data_head = df_an.head()
df_an2.to_csv('mediation_data.csv')
# 检查数据中的不同类别的数量
mbs4c_categories = df_an['mbs4c1'].nunique()
covid2_categories = df_an['covid2'].nunique()

# %%
data = df_an2
exclude_columns = ['patient_id','PLT_score', 'mbs4c1', 'covid2']  # 不包括在协变量中的列
covariates = [col for col in data.columns if col not in exclude_columns]
covariates_formula = ' + '.join(covariates)

# 第一步：分析 X 到 M 的影响
formula = f'PLT_score ~ mbs4c1+ {covariates_formula}'
model_x_to_m = smf.ols(formula, data=data).fit()

# 第二步：分析 M 到 Y 的影响，同时控制 X
formula = f'covid2 ~ PLT_score + mbs4c1+ {covariates_formula}'
model_m_to_y_controlling_x = smf.logit(formula, data=data).fit()

# 第三步：分析 X 到 Y 的直接效应（不包含中介变量 M）
formula = f'covid2 ~ mbs4c1+ {covariates_formula}'
model_x_to_y_direct = smf.logit(formula, data=data).fit()
model1_summary = model_x_to_m.summary()
model2_summary = model_m_to_y_controlling_x.summary()
model3_summary = model_x_to_y_direct.summary()
print(model1_summary,model2_summary,model3_summary)


