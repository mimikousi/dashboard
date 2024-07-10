import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
import seaborn as sns
sns.set(font='IPAexGothic')
import streamlit as st
import datetime

#脱ブタン塔のプロセスデータを読み込む
df = pd.read_csv('debutanizer_data.csv', header=0)

#時系列データなので、実務データを想定しindexに時刻を割り当てる
# 開始日時を指定
start_datetime = '2024-01-01 00:00:00'
# DataFrameの長さを取得
n = len(df)
# 日時インデックスを生成（1分間隔）
date_index = pd.date_range(start=start_datetime, periods=n, freq='T')
# DataFrameのインデックスを新しい日時インデックスに設定
df.index = date_index

# 目的変数の測定時間を考慮（5分間）
df['y'] = df['y'].shift(5)

#yがnanとなる期間のデータを削除
df = df.dropna()

dates = df.index

# Streamlitアプリケーションの作成
st.title("時系列データの相関チェック")

# 全カラム対全カラムの相関係数のヒートマップを作成
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
ax.set_title('相関係数ヒートマップ')
st.pyplot(fig)

# Streamlitアプリケーションの作成
st.title("時間遅れ変数の相関チェック")

# カラム選択
columns = df.columns.tolist()
target_column = st.selectbox("相関係数を計算するカラムを選択してください", columns)
delay_range = st.slider("時間遅れ変数の範囲を選択してください（例：0から9）", min_value=0, max_value=30, value=(0, 9))

# 遅れ変数の作成と相関係数の計算
correlations = {}
for col in columns:
    if col != target_column:
        correlations[col] = []
        for lag in range(delay_range[0], delay_range[1] + 1):
            df[f'{col}_delay_{lag}'] = df[col].shift(lag)
        df_lagged = df.dropna()

        for lag in range(delay_range[0], delay_range[1] + 1):
            correlation = df_lagged[target_column].corr(df_lagged[f'{col}_delay_{lag}'])
            correlations[col].append((lag, correlation))

# カラムごとにグラフを作成
for col in correlations:
    lagged_df = pd.DataFrame(correlations[col], columns=['Lag', 'Correlation'])

    fig, ax = plt.subplots(figsize=(10, 5))
    lagged_df.plot(kind='bar', x='Lag', y='Correlation', ax=ax, legend=False)
    ax.set_title(f'{target_column} と {col} の時間遅れ変数の相関係数')
    ax.set_ylabel('相関係数')
    ax.set_xlabel('時間遅れ')

    # グラフをStreamlitに表示
    st.pyplot(fig)
