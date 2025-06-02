# =============================================================================
# XGBoost追加特徴量モデル v11
# 
# 機能概要:
# - PCA埋め込み特徴量の生成と集約
# - 購入比率による特徴量エンジニアリング  
# - 非負値行列分解（NMF）による次元削減特徴量
# - Target Encodingによるカテゴリ変数の数値化
# - 4-fold Cross Validationによるモデル評価
# - CV: 0.7930 / LB: 0.7666
# =============================================================================

# ====================================================
# ライブラリのインポート
# ====================================================
import os
import gc
import warnings
warnings.filterwarnings('ignore')
import random
from scipy import sparse as sp
import numpy as np
import pandas as pd
import polars as pl  # 高速データ処理ライブラリ
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm
from datetime import datetime

import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD, NMF  # 次元削減手法
import lightgbm as lgb
import xgboost as xgb
import time

from scipy.sparse import coo_matrix
from pprint import pprint

# ====================================================
# 設定クラス
# ====================================================
class CFG:
    """実験設定を管理するクラス"""
    VER = 11  # 実験バージョン
    AUTHOR = 'nogawanogawa'
    COMPETITION = 'atmacup19'
    MAIN_PATH = '/content/drive/MyDrive/atmaCup19'  # パスを適宜変更
    DATA_PATH = Path(f'{MAIN_PATH}')
    OUTPUT_PATH = Path(f'{MAIN_PATH}/output')
    MODEL_PATH = Path(f'{MAIN_PATH}/model')
    OOF_DATA_PATH = Path(f'{MAIN_PATH}/oof')
    MODEL_DATA_PATH = Path(f'{MAIN_PATH}/models')
    METHOD_LIST = ['lightgbm', 'xgboost']
    METHOD_WEIGHT_DICT = {'lightgbm': 0.5, 'xgboost': 0.5}
    USE_GPU = torch.cuda.is_available()
    SEED = 42
    N_SPLIT = 4  # クロスバリデーションの分割数
    target_col_list = ['チョコレート', 'ビール', 'ヘアケア', '米（5㎏以下）']  # 予測対象カテゴリ
    metric = 'auc'
    metric_maximize_flag = True

    # XGBoostのハイパーパラメータ
    num_boost_round = 2500
    early_stopping_round = 50
    verbose = 100

    # LightGBMのパラメータ
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 6,
        'seed': SEED,
    }

    # XGBoostのパラメータ
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 3,
        'random_state': SEED,
    }

# ====================================================
# シード値固定
# ====================================================
def seed_everything(seed):
    """再現性確保のためシード値を固定"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(CFG.SEED)

# ====================================================
# データ読み込み
# ====================================================
print("データを読み込み中...")
train_session_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_session.csv', try_parse_dates=True)
test_session_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/test_session.csv', try_parse_dates=True)
train_log_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_log.csv')
train_target_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_target.csv')
jan_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/jan.csv')

# ターゲット情報を訓練セッションデータに結合
train_session_df = pl.concat([train_session_df, train_target_df], how="horizontal")

# ====================================================
# データ前処理: 返品データの除外
# ====================================================
def remove_negative_purchase(train_session_df, train_log_df, jan_df):
    """
    返品されているセッションを除外する前処理
    
    Args:
        train_session_df: 訓練セッションデータ
        train_log_df: 購買ログデータ  
        jan_df: JANコードマスタ
    
    Returns:
        tuple: 処理済みの(ログデータ, セッションデータ)
    """
    # 売上数量が0以下（返品）のデータを抽出
    neg_train_log_df = train_log_df.filter(pl.col("売上数量") <= 0)
    
    # 正の売上数量のみを残す
    train_log_df = train_log_df.filter(pl.col("売上数量") > 0)

    # 返品されているセッションをターゲットからも除外
    # 対象カテゴリでの返品があるセッションIDを特定
    ignore_sessions = neg_train_log_df.join(jan_df, on="JAN").filter(
        pl.col("カテゴリ名").is_in(CFG.target_col_list)
    )[["session_id"]]

    # 返品セッションを除外
    train_session_df = train_session_df.join(ignore_sessions, on="session_id", how="anti")

    return train_log_df, train_session_df

# 返品データ除外を適用
train_log_df, train_session_df = remove_negative_purchase(train_session_df, train_log_df, jan_df)

# ====================================================
# 期間分割の設定
# ====================================================
print("期間分割を設定中...")

# 入力期間（特徴量作成用）
# 訓練時: 7-9月、テスト時: 8-10月の購買データを使用
train_input_session_df = train_session_df.filter(
    pl.col("売上日").is_between(datetime(2024, 7, 1), datetime(2024, 9, 30))
)
test_input_session_df = train_session_df.filter(
    pl.col("売上日").is_between(datetime(2024, 8, 1), datetime(2024, 10, 31))
)

# 予測対象期間（訓練時のバリデーション用）
# 10月の購買データを予測対象とする
train_target_session_df = train_session_df.filter(
    pl.col("売上日").is_between(datetime(2024, 10, 1), datetime(2024, 10, 30))
)

# ====================================================
# クロスバリデーション用のFold分割
# ====================================================
def attach_fold(df) -> pl.DataFrame:
    """
    StratifiedGroupKFoldでFold分割を行う
    
    Args:
        df: 分割対象のデータフレーム
    
    Returns:
        pl.DataFrame: fold列が追加されたデータフレーム
    """
    # 複数カテゴリの購入パターンを組み合わせて層化変数を作成
    # ビット演算で各カテゴリの購入状況を1つの値に圧縮
    fold_df = df.with_columns(
        target = pl.col("チョコレート") * 1 + pl.col("ビール") * 2 + 
                pl.col("ヘアケア") * 4 + pl.col("米（5㎏以下）") * 8
    )

    # 顧客CDでグループ化しつつ、ターゲット分布を均等にする層化分割
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_SPLIT)

    # 各foldのセッションIDとfold番号を生成
    folds = [
        fold_df[test_index].select("session_id").with_columns(
            pl.lit(i).alias("fold")
        ) for i, (train_index, test_index) in enumerate(sgkf.split(fold_df, fold_df["target"], fold_df["顧客CD"]))
    ]

    return df.join(pl.concat(folds), on="session_id", how="left")

# バリデーション用データにfold情報を付与
train_target_session_df = attach_fold(train_target_session_df)

# ====================================================
# 特徴量エンジニアリング: 集約データの準備
# ====================================================
print("特徴量エンジニアリング: 基本集約データの作成...")

# 売上金額ベースの集約（セッション×カテゴリ単位）
price_df = train_log_df.join(
          jan_df, on="JAN", how="left"
    ).group_by(["session_id", "カテゴリ名"]).agg(
        pl.col('売上金額').sum().alias('売上金額')
    )

# 売上数量ベースの集約（セッション×カテゴリ単位）    
amount_df = train_log_df.join(
          jan_df, on="JAN", how="left"
    ).group_by(["session_id", "カテゴリ名"]).agg(
        pl.col('売上数量').sum().alias('売上数量')
    )

# ====================================================
# 非負値行列分解（NMF）特徴量の生成
# ====================================================
def create_NMF_feature(df, train_input_session_df, test_input_session_df, 
                       row, col, value):
    """
    NMF（Non-negative Matrix Factorization）で次元削減した特徴量を生成
    
    Args:
        df: 集約済みデータ（価格or数量）
        train_input_session_df: 訓練用入力セッション
        test_input_session_df: テスト用入力セッション
        row: 行の軸（顧客CD）
        col: 列の軸（カテゴリ名）
        value: 値の軸（売上金額or売上数量）
    
    Returns:
        tuple: (訓練用NMF特徴量, テスト用NMF特徴量)
    """
    # 全カテゴリのリストを取得
    all_cols = df[col].unique().to_list()
    
    # 訓練データとテストデータをピボット形式に変換
    train_df = train_input_session_df.join(df, on="session_id", how='left')[[row, col, value]]
    test_df = test_input_session_df.join(df, on="session_id", how='left')[[row, col, value]]

    # 顧客×カテゴリのマトリックスに変換
    train_df = train_df.group_by([row, col]).agg(
        pl.col(value).sum()
    ).pivot(
        values=value,
        index=row,
        columns=col,
        aggregate_function="sum"
    )
    
    test_df = test_df.group_by([row, col]).agg(
        pl.col(value).sum()
    ).pivot(
        values=value,
        index=row,
        columns=col,
        aggregate_function="sum"
    )

    # カラム名を全カテゴリに揃える（欠損カテゴリは0で埋める）
    add_cols = list(set(all_cols).difference(set(train_df.columns)))
    train_users = train_df[row]
    train_df = train_df.with_columns(
        pl.lit(0).alias(col) for col in add_cols
    ).select(all_cols)

    # NMFモデルの学習（32次元に圧縮）
    model = NMF(n_components=32, init='random', random_state=CFG.SEED, verbose=True)
    _train_df = model.fit_transform(train_df.fill_null(0).to_numpy())

    # 訓練データの結果をデータフレーム化
    _train_df = pl.DataFrame(_train_df)
    _train_df.columns = [f"nmf_{value}_{i}" for i in range(32)]
    _train_df = _train_df.with_columns(train_users.alias(row))

    # テストデータに同じ変換を適用
    add_cols = list(set(all_cols).difference(set(test_df.columns)))
    test_users = test_df[row]
    test_df = test_df.with_columns(
        pl.lit(0).alias(col) for col in add_cols
    ).select(all_cols)

    _test_df = model.transform(test_df.fill_null(0).to_numpy())
    
    _test_df = pl.DataFrame(_test_df)
    _test_df.columns = [f"nmf_{value}_{i}" for i in range(32)]
    _test_df = _test_df.with_columns(test_users.alias(row))

    return _train_df, _test_df

print("NMF特徴量を生成中...")

# 売上金額ベースのNMF特徴量
train_price_nmf_df, test_price_nmf_df = create_NMF_feature(
    price_df, train_input_session_df, test_input_session_df,
    "顧客CD", "カテゴリ名", "売上金額"
)

# 売上数量ベースのNMF特徴量
train_amount_nmf_df, test_amount_nmf_df = create_NMF_feature(
    amount_df, train_input_session_df, test_input_session_df,
    "顧客CD", "カテゴリ名", "売上数量"
)

# ====================================================
# PCA埋め込み特徴量の生成
# ====================================================
def pca_emb(df, row, col, value, dim=32):
    """
    セッション×カテゴリの購買データをPCAで次元削減
    
    Args:
        df: 入力データ
        row: セッションID
        col: カテゴリ名
        value: 売上数量/金額
        dim: 圧縮次元数
    
    Returns:
        pl.DataFrame: PCA特徴量
    """
    # ラベルエンコーディング
    row_le = LabelEncoder()
    col_le = LabelEncoder()

    df = df.with_columns(
        pl.Series(col_le.fit_transform(df[col])).alias("col"),
        pl.Series(row_le.fit_transform(df[row])).alias("row"),
    )

    # スパース行列の作成
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    data = df[value].to_numpy()
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1))

    # TruncatedSVD（PCA）で次元圧縮
    clf = TruncatedSVD(dim, random_state=CFG.SEED)
    Xpca = clf.fit_transform(matrix)

    # セッションIDごとの特徴量を取得
    row_ids = df[row].unique()
    d = pl.from_pandas(
        pd.DataFrame([Xpca[row] for row in row_le.transform(row_ids)])
    ).with_columns(
        pl.all().name.prefix(f"{col}_{value}_pca_")
    ).with_columns(
        pl.Series(row_ids).alias(row)
    )

    return d[[row] + [f"{col}_{value}_pca_{i}" for i in range(dim)]]

def create_pca_feature(df, input_session_df, row="session_id", col="カテゴリ名", value="売上数量"):
    """
    PCA特徴量を各属性（顧客、性別、年代、店舗）で集約
    
    Args:
        df: 購買データ
        input_session_df: セッションデータ
        row: 行軸
        col: 列軸
        value: 値軸
    
    Returns:
        tuple: 各属性別のPCA特徴量
    """
    # セッション単位のPCA特徴量を生成
    pca_emb_df = pca_emb(df, row, col, value)

    # 顧客CD別の平均特徴量
    customer_pca_df = input_session_df.join(
        pca_emb_df, on="session_id", how="left"
    ).select(
      pl.col("顧客CD"),
      pl.col([f"{col}_{value}_pca_{i}" for i in range(32)]).name.prefix("顧客CD_")
    ).group_by("顧客CD").mean()

    # 性別別の平均特徴量
    sex_pca_df = input_session_df.join(
        pca_emb_df, on="session_id", how="left"
    ).select(
      pl.col("性別"),
      pl.col([f"{col}_{value}_pca_{i}" for i in range(32)]).name.prefix("性別_")
    ).group_by("性別").mean()

    # 年代別の平均特徴量
    age_pca_df = input_session_df.join(
        pca_emb_df, on="session_id", how="left"
    ).select(
      pl.col("年代"),
      pl.col([f"{col}_{value}_pca_{i}" for i in range(32)]).name.prefix("年代_")
    ).group_by("年代").mean()

    # 店舗別の平均特徴量
    shop_pca_df = input_session_df.join(
        pca_emb_df, on="session_id", how="left"
    ).select(
      pl.col("店舗名"),
      pl.col([f"{col}_{value}_pca_{i}" for i in range(32)]).name.prefix("店舗名_")
    ).group_by("店舗名").mean()

    return customer_pca_df, sex_pca_df, age_pca_df, shop_pca_df

print("PCA特徴量を生成中...")

# 売上金額ベースのPCA特徴量
customer_pca_price_df, sex_pca_price_df, age_pca_price_df, shop_pca_price_df = create_pca_feature(
    price_df, train_input_session_df, value="売上金額"
)

# 売上数量ベースのPCA特徴量
customer_pca_amount_df, sex_pca_amount_df, age_pca_amount_df, shop_pca_amount_df = create_pca_feature(
    amount_df, train_input_session_df, value="売上数量"
)

# ====================================================
# 購入比率特徴量の生成
# ====================================================
def create_category_matrix(train_log_df, jan_df):
    """
    セッション×カテゴリの購入有無マトリックスを作成
    
    Args:
        train_log_df: 購買ログ
        jan_df: JANコードマスタ
    
    Returns:
        pl.DataFrame: 購入有無マトリックス（0/1）
    """
    # セッション×カテゴリ単位で売上数量を集約
    category_df = train_log_df.join(
          jan_df, on="JAN", how="left"
    ).group_by(["session_id", "カテゴリ名"]).agg(
        pl.col("売上数量").sum().alias("売上数量")
    )

    # ピボットテーブルに変換
    category_matrix_df = category_df.pivot(
        values="売上数量",
        index="session_id",
        columns="カテゴリ名",
        aggregate_function="sum"
    )

    # 売上数量を0/1に二値化（購入有無）
    category_matrix_df = category_matrix_df.with_columns([
        pl.col(col).clip(0,1)
        for col in category_matrix_df.columns if col != "session_id"
    ]).fill_null(0)

    return category_matrix_df

# 購入有無マトリックスにセッション属性を結合
purchase_ratio_df = pl.concat(
    [
        train_session_df[["session_id", "店舗名", "年代", "性別", "顧客CD"]],
        create_category_matrix(train_log_df, jan_df)[CFG.target_col_list]
    ], how="horizontal"
)

def purchase_ratio_feature(input_session_df, purchase_ratio_df):
    """
    各属性別の購入比率特徴量を生成
    
    Args:
        input_session_df: 入力セッションデータ
        purchase_ratio_df: 購入比率データ
    
    Returns:
        tuple: 各属性別の購入比率特徴量
    """
    # 顧客CD別の購入比率
    customer_purchase_ratio_df = input_session_df.join(
        purchase_ratio_df, on="session_id", how="left"
    ).select(
      pl.col("顧客CD"),
      pl.col(CFG.target_col_list).name.prefix("顧客CD_purchase_ratio_")
    ).group_by("顧客CD").mean()

    # 性別別の購入比率
    sex_purchase_ratio_df = input_session_df.join(
        purchase_ratio_df, on="session_id", how="left"
    ).select(
      pl.col("性別"),
      pl.col(CFG.target_col_list).name.prefix("性別_purchase_ratio_")
    ).group_by("性別").mean()

    # 年代別の購入比率
    age_purchase_ratio_df = input_session_df.join(
        purchase_ratio_df, on="session_id", how="left"
    ).select(
      pl.col("年代"),
      pl.col(CFG.target_col_list).name.prefix("年代_purchase_ratio_")
    ).group_by("年代").mean()

    # 店舗別の購入比率
    shop_purchase_ratio_df = input_session_df.join(
        purchase_ratio_df, on="session_id", how="left"
    ).select(
      pl.col("店舗名"),
      pl.col(CFG.target_col_list).name.prefix("店舗名_purchase_ratio_")
    ).group_by("店舗名").mean()

    return customer_purchase_ratio_df, sex_purchase_ratio_df, age_purchase_ratio_df, shop_purchase_ratio_df

print("購入比率特徴量を生成中...")
customer_purchase_ratio_df, sex_purchase_ratio_df, age_purchase_ratio_df, shop_purchase_ratio_df = purchase_ratio_feature(
    train_input_session_df, purchase_ratio_df
)

# ====================================================
# カテゴリ変数特徴量の生成
# ====================================================
print("カテゴリ変数特徴量を生成中...")

# 全データ（訓練+テスト）を結合
all_df = pl.concat([train_session_df[test_session_df.columns], test_session_df])

# カテゴリ変数をダミー変数化し、時間特徴量も追加
category_feature_df = pl.concat(
    [
        all_df[["session_id", "時刻", "売上日"]], 
        all_df[["年代", "性別", "店舗名"]].to_dummies()  # ワンホットエンコーディング
    ], 
    how="horizontal"
).with_columns(
    pl.col("売上日").dt.weekday().alias("weekday"),        # 曜日（0=月曜）
    pl.col("売上日").dt.day().alias("day_of_month"),       # 月内の日付
).drop("売上日")

# ====================================================
# Target Encoding特徴量の生成
# ====================================================
def train_target_encoding(df, category_col):
    """
    訓練時のTarget Encoding（クロスバリデーション対応）
    
    Args:
        df: 訓練データ
        category_col: エンコーディング対象のカテゴリ列
    
    Returns:
        pl.DataFrame: Target Encoding済み特徴量
    """
    _target_encoded_dfs = []
    
    # 各foldでvalid以外のデータを使ってターゲット平均値を計算
    for fold in range(CFG.N_SPLIT):
        _target_encoded_df = df.filter(
          pl.col("fold") != fold  # 当該foldを除外
        ).group_by(category_col).agg(
            pl.col(CFG.target_col_list).mean()  # カテゴリ別のターゲット平均値
        ).with_columns(
            pl.col(CFG.target_col_list).name.prefix(f"target_encoding_{category_col}_"),
            pl.lit(fold).alias("fold")
        )
        _target_encoded_dfs.append(_target_encoded_df)

    return pl.concat(_target_encoded_dfs)[[category_col, "fold"] + 
                                         [f"target_encoding_{category_col}_{col}" for col in CFG.target_col_list]]

def test_target_encoding(df, category_col):
    """
    テスト時のTarget Encoding（全訓練データを使用）
    
    Args:
        df: 訓練データ
        category_col: エンコーディング対象のカテゴリ列
    
    Returns:
        pl.DataFrame: Target Encoding済み特徴量
    """
    _target_encoded_df = df.group_by(category_col).agg(
        pl.col(CFG.target_col_list).mean()
    ).with_columns(
        pl.col(CFG.target_col_list).name.prefix(f"target_encoding_{category_col}_"),
    )

    return _target_encoded_df[[category_col] + 
                              [f"target_encoding_{category_col}_{col}" for col in CFG.target_col_list]]

print("Target Encoding特徴量を生成中...")

# 訓練用Target Encoding（CV対応）
train_target_encoding_shop_df = train_target_encoding(train_target_session_df, "店舗名")
train_target_encoding_sex_df = train_target_encoding(train_target_session_df, "性別")
train_target_encoding_age_df = train_target_encoding(train_target_session_df, "年代")

# テスト用Target Encoding（全データ使用）
test_target_encoding_shop_df = test_target_encoding(train_target_session_df, "店舗名")
test_target_encoding_sex_df = test_target_encoding(train_target_session_df, "性別")
test_target_encoding_age_df = test_target_encoding(train_target_session_df, "年代")

# ====================================================
# XGBoostモデルの訓練
# ====================================================
print("XGBoostモデルの訓練を開始...")

# XGBoostのパラメータ設定
params = {
    'tree_method': "hist",                    # ヒストグラムベースの高速アルゴリズム
    'multi_strategy': "multi_output_tree",    # マルチターゲット対応
    'n_estimators': 1000,                     # ブースティング回数
    'early_stopping_rounds': 50,             # アーリーストッピング
    'learning_rate': 0.05,                   # 学習率
    'gamma': 0.1,                            # 最小分割損失
    'subsample': 0.8,                        # サンプリング比率
    'colsample_bytree': 0.3,                 # 特徴量サンプリング比率
    'min_child_weight': 3,                   # 葉ノードの最小重み
    'max_depth': 6,                          # 木の最大深度
    'seed': CFG.SEED,
}

# 予測結果とスコア保存用のリスト
pred_dfs = []  # 各foldの予測結果
true_dfs = []  # 各foldの正解ラベル

# ====================================================
# クロスバリデーション実行
# ====================================================
for fold in range(CFG.N_SPLIT):
    print(f"#### Fold: {fold} ####")
    
    # 当該foldを訓練/バリデーションに分割
    _train_target_session_df = train_target_session_df.filter(pl.col("fold") != fold)  # 訓練用
    _valid_target_session_df = train_target_session_df.filter(pl.col("fold") == fold)   # バリデーション用

    # =================
    # 特徴量結合（訓練データ）
    # =================
    train_df = _train_target_session_df.join(
        shop_pca_price_df, on="店舗名", how="left"              # 店舗別PCA特徴量（売上金額）
    ).join(
        age_pca_price_df, on="年代", how="left"                # 年代別PCA特徴量（売上金額）
    ).join(
        sex_pca_price_df, on="性別", how="left"                # 性別PCA特徴量（売上金額）
    ).join(
        customer_pca_price_df, on="顧客CD", how="left"         # 顧客別PCA特徴量（売上金額）
    ).join(
        shop_pca_amount_df, on="店舗名", how="left"            # 店舗別PCA特徴量（売上数量）
    ).join(
        age_pca_amount_df, on="年代", how="left"               # 年代別PCA特徴量（売上数量）
    ).join(
        sex_pca_amount_df, on="性別", how="left"               # 性別PCA特徴量（売上数量）
    ).join(
        customer_pca_amount_df, on="顧客CD", how="left"        # 顧客別PCA特徴量（売上数量）
    ).join(
        train_amount_nmf_df, on="顧客CD", how="left"           # NMF特徴量（売上数量）
    ).join(
        train_price_nmf_df, on="顧客CD", how="left"            # NMF特徴量（売上金額）
    ).join(
        shop_purchase_ratio_df, on="店舗名", how="left"        # 店舗別購入比率
    ).join(
        age_purchase_ratio_df, on="年代", how="left"           # 年代別購入比率
    ).join(
        sex_purchase_ratio_df, on="性別", how="left"           # 性別購入比率
    ).join(
        customer_purchase_ratio_df, on="顧客CD", how="left"    # 顧客別購入比率
    ).join(
        category_feature_df, on="session_id", how="left"       # カテゴリ変数特徴量
    ).join(
        train_target_encoding_shop_df, on=["店舗名", "fold"], how="left"  # Target Encoding（店舗）
    ).join(
        train_target_encoding_sex_df, on=["性別", "fold"], how="left"     # Target Encoding（性別）
    ).join(
        train_target_encoding_age_df, on=["年代", "fold"], how="left"     # Target Encoding（年代）
    )

    # =================
    # 特徴量結合（バリデーションデータ）
    # =================
    valid_df = _valid_target_session_df.join(
        shop_pca_price_df, on="店舗名", how="left"
    ).join(
        age_pca_price_df, on="年代", how="left"
    ).join(
        sex_pca_price_df, on="性別", how="left"
    ).join(
        customer_pca_price_df, on="顧客CD", how="left"
    ).join(
        shop_pca_amount_df, on="店舗名", how="left"
    ).join(
        age_pca_amount_df, on="年代", how="left"
    ).join(
        sex_pca_amount_df, on="性別", how="left"
    ).join(
        customer_pca_amount_df, on="顧客CD", how="left"
    ).join(
        train_amount_nmf_df, on="顧客CD", how="left"
    ).join(
        train_price_nmf_df, on="顧客CD", how="left"
    ).join(
        shop_purchase_ratio_df, on="店舗名", how="left"
    ).join(
        age_purchase_ratio_df, on="年代", how="left"
    ).join(
        sex_purchase_ratio_df, on="性別", how="left"
    ).join(
        customer_purchase_ratio_df, on="顧客CD", how="left"
    ).join(
        category_feature_df, on="session_id", how="left"
    ).join(
        train_target_encoding_shop_df, on=["店舗名", "fold"], how="left"
    ).join(
        train_target_encoding_sex_df, on=["性別", "fold"], how="left"
    ).join(
        train_target_encoding_age_df, on=["年代", "fold"], how="left"
    )

    # =================
    # 特徴量とターゲットの分離
    # =================
    # 学習に不要な列を除外
    exclude_cols = CFG.target_col_list + ["session_id", "fold", "売上日", "顧客CD", "店舗名", "性別", "年代"]
    
    X_train = train_df.drop(exclude_cols)  # 訓練用特徴量
    y_train = train_df[CFG.target_col_list]  # 訓練用ターゲット
    X_val = valid_df.drop(exclude_cols)   # バリデーション用特徴量  
    y_val = valid_df[CFG.target_col_list]   # バリデーション用ターゲット

    # =================
    # XGBoostモデルの学習
    # =================
    clf = xgb.XGBClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],  # バリデーションセットでアーリーストッピング
    )

    # モデルの保存
    best_model_path = CFG.MODEL_PATH / f"exp{CFG.VER}_xgb_fold_{fold}.model"
    clf.save_model(best_model_path)

    # =================
    # 予測実行
    # =================
    preds = clf.predict_proba(X_val)  # 各クラスの予測確率

    # 予測結果をデータフレーム化
    val_df = pd.DataFrame(preds, columns=[CFG.target_col_list])
    val_df["session_id"] = _valid_target_session_df["session_id"].to_list()

    # 結果を保存
    pred_dfs.append(val_df)
    true_dfs.append(_valid_target_session_df.to_pandas())

# ====================================================
# クロスバリデーションスコアの計算
# ====================================================
def calc_score(y_true, y_pred):
    """
    Macro AUCスコアを計算
    
    Args:
        y_true: 正解ラベル
        y_pred: 予測確率
    
    Returns:
        float: Macro AUCスコア
    """
    score_list = []
    for target_col in range(4):  # 4つのターゲットカテゴリ
        score = roc_auc_score(y_true[:, target_col], y_pred[:, target_col])
        score_list.append(score)
    return np.mean(score_list)

# 全foldの結果を統合してスコア計算
cv_score = calc_score(
    pd.concat(true_dfs)[CFG.target_col_list].values,  # 正解ラベル
    pd.concat(pred_dfs)[CFG.target_col_list].values   # 予測確率
)

print(f"クロスバリデーション Macro AUCスコア: {cv_score:.6f}")

# バリデーション予測結果をCSVとして保存
valid_preds_df = pd.concat(pred_dfs)
valid_preds_df.to_csv(CFG.OUTPUT_PATH / f"exp{CFG.VER:03d}_oof_pred.csv", index=False)
print(f"バリデーション予測結果を保存: exp{CFG.VER:03d}_oof_pred.csv")

# ====================================================
# テストデータでの推論
# ====================================================
print("テストデータでの推論を開始...")

# テスト用の特徴量を再生成（入力期間をテスト用に変更）
customer_pca_price_df, sex_pca_price_df, age_pca_price_df, shop_pca_price_df = create_pca_feature(
    price_df, test_input_session_df, value="売上金額"
)
customer_pca_amount_df, sex_pca_amount_df, age_pca_amount_df, shop_pca_amount_df = create_pca_feature(
    amount_df, test_input_session_df, value="売上数量"
)
customer_purchase_ratio_df, sex_purchase_ratio_df, age_purchase_ratio_df, shop_purchase_ratio_df = purchase_ratio_feature(
    test_input_session_df, purchase_ratio_df
)

# =================
# テストデータの特徴量結合
# =================
test_df = test_session_df.join(
      shop_pca_price_df, on="店舗名", how="left"
  ).join(
      age_pca_price_df, on="年代", how="left"
  ).join(
      sex_pca_price_df, on="性別", how="left"
  ).join(
      customer_pca_price_df, on="顧客CD", how="left"
  ).join(
      shop_pca_amount_df, on="店舗名", how="left"
  ).join(
      age_pca_amount_df, on="年代", how="left"
  ).join(
      sex_pca_amount_df, on="性別", how="left"
  ).join(
      customer_pca_amount_df, on="顧客CD", how="left"
  ).join(
      test_amount_nmf_df, on="顧客CD", how="left"           # テスト用NMF特徴量
  ).join(
      test_price_nmf_df, on="顧客CD", how="left"            # テスト用NMF特徴量
  ).join(
      shop_purchase_ratio_df, on="店舗名", how="left"
  ).join(
      age_purchase_ratio_df, on="年代", how="left"
  ).join(
      sex_purchase_ratio_df, on="性別", how="left"
  ).join(
      customer_purchase_ratio_df, on="顧客CD", how="left"
  ).join(
      category_feature_df, on="session_id", how="left"
  ).join(
      test_target_encoding_shop_df, on="店舗名", how="left"  # テスト用Target Encoding
  ).join(
      test_target_encoding_sex_df, on="性別", how="left"
  ).join(
      test_target_encoding_age_df, on="年代", how="left"
  )

# テスト用特徴量の準備
X_test = test_df.drop(["session_id", "売上日", "顧客CD", "店舗名", "性別", "年代"])

# =================
# アンサンブル予測
# =================
print("アンサンブル予測を実行中...")

clf = xgb.XGBClassifier(**params)
preds = []

# 各foldのモデルで予測を実行
for fold in range(CFG.N_SPLIT):
    best_model_path = CFG.MODEL_PATH / f"exp{CFG.VER}_xgb_fold_{fold}.model"
    clf.load_model(best_model_path)
    pred = clf.predict_proba(X_test)
    preds.append(pred)

# 全foldの予測を平均化
pred = None
for p in preds:
    if pred is None:
        pred = p
    else:
        pred = pred + p
pred = pred / len(preds)

print(f"予測形状: {pred.shape}")
print(f"予測値の例: \n{pred[:5]}")

# ====================================================
# 提出ファイルの作成
# ====================================================
print("提出ファイルを作成中...")

# 予測結果をデータフレーム化
submission = pd.DataFrame(pred, columns=CFG.target_col_list)

# 提出ファイルの保存
submission.to_csv(CFG.OUTPUT_PATH / f"exp{CFG.VER:03d}_submission.csv", index=False)
print(f"提出ファイルを保存: exp{CFG.VER:03d}_submission.csv")

# ====================================================
# 結果の確認
# ====================================================
print("\n=== 最終結果 ===")
print(f"クロスバリデーション Macro AUCスコア: {cv_score:.6f}")
print("\n予測値の統計:")
print(submission.describe())

print("\n予測値の先頭5件:")
print(submission.head())

print(f"\n各カテゴリの予測値範囲:")
for col in CFG.target_col_list:
    print(f"  {col}: {submission[col].min():.6f} ~ {submission[col].max():.6f}")

print("\n処理完了！")
