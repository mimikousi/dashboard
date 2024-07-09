import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
import seaborn as sns
sns.set(font='IPAexGothic')
import streamlit as st
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Streamlitアプリケーションの作成
st.title("時系列データのトレンド比較")

dates = df.index

# 変数の選択
columns = df.columns.tolist()
column1 = st.selectbox("比較する最初の変数を選択してください", columns)
column2 = st.selectbox("比較する2番目の変数を選択してください", columns)

# 時間遅れの選択
lag1 = st.slider("最初の変数の時間遅れを選択してください", min_value=0, max_value=30, value=0)
lag2 = st.slider("2番目の変数の時間遅れを選択してください", min_value=0, max_value=30, value=0)

# 期間選択
start_date = st.slider("開始日", min_value=dates.min().to_pydatetime(), max_value=dates.max().to_pydatetime(), value=dates.min().to_pydatetime())
end_date = st.slider("終了日", min_value=dates.min().to_pydatetime(), max_value=dates.max().to_pydatetime(), value=dates.max().to_pydatetime())


# 選択された期間のデータをフィルタリング
filtered_df = df.loc[start_date:end_date]

# 時間遅れ変数の生成
lagged_column1 = filtered_df[column1].shift(lag1)
lagged_column2 = filtered_df[column2].shift(lag2)

# 時系列トレンドグラフを作成
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()
ax1.plot(filtered_df.index, lagged_column1, label=f'{column1} (lag {lag1})')
ax2.plot(filtered_df.index, lagged_column2, label=f'{column2} (lag {lag2})', color='orange')
ax1.set_title(f'{column1} と {column2} の時系列トレンド')
#ax1.set_xlabel('Date')
ax1.set_ylabel(f'{column1} (lag {lag1})')
ax2.set_ylabel(f'{column2} (lag {lag2})')
plt.grid(False)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='best')

# グラフをStreamlitに表示
st.pyplot(fig)

# 相関係数を計算
correlation = lagged_column1.corr(lagged_column2)

# 相関係数を表示
st.write(f"選択された期間における {column1} と {column2} の相関係数: {correlation:.2f}")


# データの対応する行が存在しないデータを除外
lagged_df = pd.DataFrame({f'{column1}_lag{lag1}': lagged_column1, f'{column2}_lag{lag2}': lagged_column2}).dropna()

# 散布図と線形近似直線を作成
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=lagged_df[f'{column1}_lag{lag1}'], y=lagged_df[f'{column2}_lag{lag2}'], ax=ax)
ax.set_title(f'{column1} (lag {lag1}) と {column2} (lag {lag2}) の散布図')
ax.set_xlabel(f'{column1} (lag {lag1})')
ax.set_ylabel(f'{column2} (lag {lag2})')

# 線形回帰モデルを作成
model = LinearRegression()
X = lagged_df[f'{column1}_lag{lag1}'].values.reshape(-1, 1)
y = lagged_df[f'{column2}_lag{lag2}'].values.reshape(-1, 1)
model.fit(X, y)
line_x = np.linspace(X.min(), X.max(), 100)
line_y = model.predict(line_x.reshape(-1, 1))

# 近似直線をプロット
ax.plot(line_x, line_y, color='red', label='Linear Fit')

# 近似直線の式と相関係数を図に追加
slope = model.coef_[0][0]
intercept = model.intercept_[0]
standard_error = np.sqrt(mean_squared_error(y, model.predict(X)))

ax.text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.05, 0.90, f'R= {correlation:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.text(0.05, 0.85, f'RMSE= {standard_error:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# 凡例を追加
#ax.legend()

# グラフをStreamlitに表示
st.pyplot(fig)
