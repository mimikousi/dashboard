import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

# データの読み込み
#@st.cache_data
@st.cache
def load_data():
    df = pd.read_csv('debutanizer_data.csv', header=0)
    start_datetime = '2024-01-01 00:00:00'
    n = len(df)
    date_index = pd.date_range(start=start_datetime, periods=n, freq='T')
    df.index = date_index
    df['y'] = df['y'].shift(5)
    df = df.dropna()
    return df

df = load_data()

# 説明変数Xと目的変数yに分割
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# アプリケーションのメインタイトル
st.title('PCAとt-SNEの可視化')

# サイドバーのオプション
variable = st.sidebar.selectbox('プロットを色分けする変数を選択:', df.columns)
perplexity_value = st.sidebar.slider('t-SNE Perplexity', min_value=5, max_value=50, value=30)

# データの標準化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

# 主成分分析 (PCA)
n_components = X.shape[1]  # 列数を取得
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_scaled)
pca_df_ = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
pca_df_.index = X.index  # 元の日時インデックスを保持
pca_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])
pca_df[variable] = df[variable].values

# PCAの散布図のタイトル
st.header('PCA 散布図')

# PCA散布図
fig, ax = plt.subplots()
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=variable, ax=ax)
st.pyplot(fig)

# 寄与率と累積寄与率のグラフのタイトル
st.header('寄与率と累積寄与率')

# 寄与率の表示
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='b')
ax2.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), color='r')
ax1.set_xlabel('主成分')
ax1.set_ylabel('説明される分散（寄与率）')
ax2.set_ylabel('累積寄与率')
ax2.grid(False)
st.pyplot(fig)

# t-SNEの実行
tsne = TSNE(n_components=2, random_state=0,perplexity=perplexity_value)
tsne_results = tsne.fit_transform(df_scaled)
tsne_df = pd.DataFrame(data=tsne_results, columns=['D1', 'D2'])
tsne_df[variable] = df[variable].values

# t-SNEの散布図のタイトル
st.header('t-SNE 可視化')

# t-SNE散布図
fig, ax = plt.subplots()
sns.scatterplot(data=tsne_df, x='D1', y='D2', hue=variable, ax=ax)
st.pyplot(fig)
