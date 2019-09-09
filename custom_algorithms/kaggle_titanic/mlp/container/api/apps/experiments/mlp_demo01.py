# %% [markdown]
# # PyTorchでMLPをデモってみる
# WIP (以下落書きでしかない)
# %%
import pandas as pd
# from matplotlib.pyplot as plt

# %%
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# %%
train_df.head()

#%%
test_df.head()

# %% [markdown]
# ## Cleaning

# %%
# 重複ID，欠損等の確認
train_df.info()
# %%
test_df.info()

# %% [markdown]
# 重複，欠損
# -> Aage, Fare, Cabinにあり

# %% [markdown]
# TODO: 重複サンプル確認

# %% [markdown]
# ## 基本分析
# ### クラス不均衡チェック

# %%
survived_counts = train_df['Survived'].value_counts()
survived_counts.plot.barh()

# %% [markdown]
# 1.6倍以上の開き

# %% [markdown]
# **特徴エンジニアリング終了後，Negative Down Samplingでサンプル数を均衡**
# Samplingが偏らないようにしたいが，，

# %% [markdown]
# ### 説明変数の取捨選択
# %% [markdown]
# #### カテゴリカル変数化
# とりあえず欠損値は適当なカテゴリを割り振ってやる

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

targets = ['Sex', 'Ticket', 'Cabin', 'Embarked']
train_test_df = train_df.append(test_df).reset_index(drop=True)
train_test_df[targets] = train_test_df[targets].fillna('NaN')
train_test_df[targets] = train_test_df[targets].apply(le.fit_transform)

train_df = train_test_df[:train_df.shape[0]]
test_df = train_test_df[train_df.shape[0]:].reset_index(drop=True)
# %%
train_df
# %%
test_df

# %% [markdown]
# #### Non Tree Based modelに適用するのでOne-hot Encoding化したい

# %%
# カテゴリ数を調査
train_test_df.nunique()
# %%
# to_onehot
from scipy.sparse import hstack
targets = ['Sex', 'Ticket', 'Cabin', 'Embarked']
targets += ['Pclass']
ohe = preprocessing.OneHotEncoder()

one_hotted = None
for i, target in enumerate(targets):
    target_inp = train_test_df[target].values.reshape(-1, 1)
    if i == 0:
        one_hotted = ohe.fit_transform(target_inp)
    else:
        tmp = ohe.fit_transform(target_inp)
        one_hotted = hstack([one_hotted, tmp])

# %%
one_hotted
# %% [markdown]

# #### 相関をみる

# %%

df_corr = train_df[train_df.columns[1:]].corr(method='spearman')
df_corr

# %%
df_corr['Survived']

# %%
import seaborn as sns
sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)

# %%
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = \
    sampler.fit(train_df['Survived'], train_df[train_df.columns[2:]])
# %%


#%%
