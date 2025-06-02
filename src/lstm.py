# -*- coding: utf-8 -*-
"""
LSTM006c - 時系列購買予測モデル（完全データ分離版）
顧客の購買履歴を用いて、次の購買カテゴリを予測するLSTMモデル
"""

# ========================================
# ライブラリのインポート
# ========================================
import os
import random
import math
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 設定クラス - モデルのハイパーパラメータと設定
# ========================================
class CFG:
    # データパス設定
    COMPETITION = 'atmacup19'
    MAIN_PATH = '/content/drive/MyDrive/atmaCup19'  # Googleドライブのパス
    DATA_PATH = Path(f'{MAIN_PATH}')
    OUTPUT_PATH = Path(f'{MAIN_PATH}/output')
    MODEL_PATH = Path(f'{MAIN_PATH}/model')
    OOF_DATA_PATH = Path(f'{MAIN_PATH}/oof')
    MODEL_DATA_PATH = Path(f'{MAIN_PATH}/models')

    # モデルのハイパーパラメータ
    SEED = 42                  # 乱数シード（再現性確保）
    NUM_SESSIONS = 5           # 使用する過去セッション数（session-4からsession0まで）
    EMBED_DIM = 32             # SVDで圧縮する次元数
    HIDDEN_DIM = 64            # LSTMの隠れ層次元数
    BATCH_SIZE = 1024          # バッチサイズ
    EPOCHS = 30                # 最大エポック数
    LEARNING_RATE = 1e-4       # 学習率
    WEIGHT_DECAY = 0.0001      # L2正則化係数
    RANDOM_CUT_PROB = 0.3      # データ拡張用のランダムカット確率
    
    # 予測対象カテゴリ
    CATEGORIES = ['チョコレート', 'ビール', 'ヘアケア', '米（5㎏以下）']
    NUM_CATEGORIES = len(CATEGORIES)

    # 検証パラメータ
    VALID_SIZE = 0.2                # ホールドアウト検証用の割合
    EARLY_STOPPING_PATIENCE = 5     # 早期停止の忍耐回数

    # 保存設定
    SAVE_OOF = True            # Out of Fold予測の保存
    SAVE_TEST_PREDS = True     # テスト予測の保存

    # デバッグモード設定
    DEBUG_MODE = False         # デバッグ時はTrueに変更
    DEBUG_SAMPLE_SIZE = 2000   # デバッグ時のサンプル数


def seed_everything(seed=42):
    """
    再現性確保のための乱数シード固定
    
    Args:
        seed (int): 固定する乱数シード値
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ========================================
# データ読み込み・分割関数
# ========================================
def load_data():
    """
    データの読み込みと時系列での分割を行う
    
    Returns:
        tuple: 分割されたデータセット
        - 訓練データ: ~2024-09-30
        - 検証データ: 2024-10-01 ~ 2024-10-31
        - テストデータ: 2024-11-01~
    """
    # CSVファイルの読み込み
    train_session_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_session.csv', try_parse_dates=True)
    test_session_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/test_session.csv', try_parse_dates=True)
    train_log_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_log.csv')
    train_target_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/train_target.csv')
    jan_df = pl.read_csv(CFG.DATA_PATH / 'atmacup19_dataset/jan.csv')

    # テストセッションの元々の順序を保存（提出時に必要）
    original_test_session_order = test_session_df['session_id'].to_list()

    # データの日付範囲を確認
    print(f"訓練セッションデータの日付範囲: {train_session_df['売上日'].min()} から {train_session_df['売上日'].max()}")
    print(f"テストセッションデータの日付範囲: {test_session_df['売上日'].min()} から {test_session_df['売上日'].max()}")

    # 時系列分割のための日付境界設定
    train_end_date = pd.Timestamp('2024-09-30')    # 訓練データの終了日
    valid_end_date = pd.Timestamp('2024-10-31')    # 検証データの終了日

    print(f"時系列分割: 訓練データ〜{train_end_date.strftime('%Y-%m-%d')}, 検証データ〜{valid_end_date.strftime('%Y-%m-%d')}, それ以降はテスト")

    # train_targetにsession_idを追加（必要な場合）
    if 'session_id' not in train_target_df.columns:
        print("train_target.csvにsession_idが含まれていません。train_session.csvから追加します。")
        if len(train_target_df) == len(train_session_df):
            train_target_df = train_target_df.with_columns(
                pl.lit(train_session_df['session_id']).alias('session_id')
            )
            print(f"session_idを{len(train_target_df)}行追加しました。")
        else:
            raise ValueError(f"train_targetとtrain_sessionの行数が一致しません: {len(train_target_df)} vs {len(train_session_df)}")

    # 日付に基づく時系列分割
    train_session_pd = train_session_df.to_pandas()
    train_session_pd['売上日'] = pd.to_datetime(train_session_pd['売上日'])

    # 訓練データと検証データに分割
    actual_train_session = train_session_pd[train_session_pd['売上日'] <= train_end_date].copy()
    valid_session = train_session_pd[(train_session_pd['売上日'] > train_end_date) &
                                     (train_session_pd['売上日'] <= valid_end_date)].copy()

    # polarsに戻す
    actual_train_session_df = pl.from_pandas(actual_train_session)
    valid_session_df = pl.from_pandas(valid_session)

    print(f"実際の訓練データセッション数: {len(actual_train_session_df)}")
    print(f"検証データセッション数: {len(valid_session_df)}")
    print(f"テストデータセッション数: {len(test_session_df)}")

    # セッションIDの重複チェック（データリーク防止）
    train_session_ids = set(actual_train_session_df['session_id'].to_list())
    valid_session_ids = set(valid_session_df['session_id'].to_list())
    test_session_ids = set(test_session_df['session_id'].to_list())

    # 重複があれば警告
    train_valid_overlap = train_session_ids.intersection(valid_session_ids)
    train_test_overlap = train_session_ids.intersection(test_session_ids)
    valid_test_overlap = valid_session_ids.intersection(test_session_ids)

    if train_valid_overlap:
        print(f"警告: 訓練データと検証データの間に{len(train_valid_overlap)}個のセッションIDの重複があります。")
    if train_test_overlap:
        print(f"警告: 訓練データとテストデータの間に{len(train_test_overlap)}個のセッションIDの重複があります。")
    if valid_test_overlap:
        print(f"警告: 検証・テストデータ間に{len(valid_test_overlap)}個のセッションID重複があります。")

    # エンコーダ情報の取得
    train_customers = encoders_info['train_customers']
    valid_customers = encoders_info['valid_customers']
    test_customers = encoders_info['test_customers']

    # 顧客セッションの作成（データリークを防ぐため各データセットごとに独立して処理）
    print("顧客ごとのセッションリストを作成しています...")

    # 1. 訓練データの顧客セッション
    train_customer_sessions = defaultdict(list)
    train_customer_groups = train_data.groupby('顧客CD')

    for customer_id, group in tqdm(train_customer_groups, desc="訓練データの顧客セッション"):
        sorted_group = group.sort_values(['売上日', '時刻'])
        train_customer_sessions[customer_id] = sorted_group['session_id'].tolist()

    # 2. 検証データの顧客セッション（訓練期間との連携はまだ行わない）
    valid_customer_sessions = defaultdict(list)
    valid_customer_groups = valid_data.groupby('顧客CD')

    for customer_id, group in tqdm(valid_customer_groups, desc="検証データの顧客セッション"):
        sorted_group = group.sort_values(['売上日', '時刻'])
        valid_customer_sessions[customer_id] = sorted_group['session_id'].tolist()

    # ========================================
    # 金銭関連特徴の計算
    # ========================================
    print("セッション・顧客単位の金銭関連特徴を計算しています...")

    # セッションIDと顧客CDのマッピング辞書
    session_to_customer = dict(zip(train_data['session_id'], train_data['顧客CD']))
    # 検証データのマッピングも追加
    session_to_customer.update(dict(zip(valid_data['session_id'], valid_data['顧客CD'])))
    # テストデータのマッピングも追加
    session_to_customer.update(dict(zip(test_data['session_id'], test_data['顧客CD'])))

    # 訓練データのみから金銭関連特徴を抽出
    train_log_filtered = train_log[train_log['session_id'].isin(train_session_ids)].copy()

    # 金銭関連カラムの検出
    money_cols = [col for col in train_log_filtered.columns if any(keyword in col.lower() for keyword in
                 ['金額', '値引', '値割', '割引', '売上金額', '売価', '支払い', '支払額', 'price', 'discount', 'payment', 'amount'])]

    # 金銭関連カラムがない場合のフォールバック
    if not money_cols:
        print("警告: 金銭関連カラムが見つかりませんでした。代替カラムを使用します。")
        # JAN単位の行から適切な代替カラムを探す
        potential_cols = [col for col in train_log_filtered.columns
                         if col not in ['session_id', 'JAN', 'mapped_category']]

        if potential_cols:
            money_cols = potential_cols[:min(3, len(potential_cols))]
            print(f"代替金銭カラム: {money_cols}")

    print(f"使用する金銭関連カラム: {money_cols}")

    # 訓練データのセッション単位の金銭集計
    train_session_monetary_stats = {}

    if money_cols:
        # セッションごとの金銭特徴の集計
        train_money_agg = train_log_filtered.groupby('session_id')[money_cols].agg(['sum', 'mean', 'count']).reset_index()
        train_money_agg.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
                                  for col in train_money_agg.columns]

        # セッションの金銭特徴辞書を作成
        for _, row in train_money_agg.iterrows():
            session_id = row['session_id']
            train_session_monetary_stats[session_id] = {col: row[col] for col in train_money_agg.columns if col != 'session_id'}

    # 検証データの金銭特徴を計算
    valid_session_monetary_stats = {}

    # 検証データのログを使用する場合
    valid_log_filtered = train_log[train_log['session_id'].isin(valid_session_ids)].copy()

    if len(valid_log_filtered) > 0 and money_cols:
        # 検証セッションごとの金銭特徴の集計
        valid_money_agg = valid_log_filtered.groupby('session_id')[money_cols].agg(['sum', 'mean', 'count']).reset_index()
        valid_money_agg.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}"
                                  for col in valid_money_agg.columns]

        # 検証セッションの金銭特徴辞書を作成
        for _, row in valid_money_agg.iterrows():
            session_id = row['session_id']
            valid_session_monetary_stats[session_id] = {col: row[col] for col in valid_money_agg.columns if col != 'session_id'}

        print(f"検証セッションの金銭特徴を計算: {len(valid_session_monetary_stats)}件")

    # すべてのセッション金銭特徴を統合
    session_monetary_stats = {**train_session_monetary_stats, **valid_session_monetary_stats}

    # 顧客単位の金銭集計（訓練データのみ）
    customer_monetary_stats = defaultdict(dict)
    customer_counts = defaultdict(int)

    # 訓練データからの顧客統計を計算
    for session_id, stats in train_session_monetary_stats.items():
        customer_id = session_to_customer.get(session_id)
        if customer_id:
            # 顧客単位で統計量を蓄積
            for key, value in stats.items():
                if key in customer_monetary_stats[customer_id]:
                    customer_monetary_stats[customer_id][key] += value
                else:
                    customer_monetary_stats[customer_id][key] = value
            customer_counts[customer_id] += 1

    # 顧客単位の金銭集計（訓練+検証データ）
    customer_monetary_stats_with_valid = defaultdict(dict)
    customer_counts_with_valid = defaultdict(int)

    # 訓練データからの顧客統計を計算
    for session_id, stats in train_session_monetary_stats.items():
        customer_id = session_to_customer.get(session_id)
        if customer_id:
            # 顧客単位で統計量を蓄積
            for key, value in stats.items():
                if key in customer_monetary_stats_with_valid[customer_id]:
                    customer_monetary_stats_with_valid[customer_id][key] += value
                else:
                    customer_monetary_stats_with_valid[customer_id][key] = value
            customer_counts_with_valid[customer_id] += 1

    # 検証データからの顧客統計を計算
    for session_id, stats in valid_session_monetary_stats.items():
        customer_id = session_to_customer.get(session_id)
        if customer_id:
            # 顧客単位で統計量を蓄積
            for key, value in stats.items():
                if key in customer_monetary_stats_with_valid[customer_id]:
                    customer_monetary_stats_with_valid[customer_id][key] += value
                else:
                    customer_monetary_stats_with_valid[customer_id][key] = value
            customer_counts_with_valid[customer_id] += 1

    # 顧客単位で平均を計算（訓練データのみ）
    for customer_id, stats in customer_monetary_stats.items():
        count = customer_counts[customer_id]
        for key in stats:
            stats[key] = stats[key] / count if count > 0 else 0

    # 顧客単位で平均を計算（訓練+検証データ）
    for customer_id, stats in customer_monetary_stats_with_valid.items():
        count = customer_counts_with_valid[customer_id]
        for key in stats:
            stats[key] = stats[key] / count if count > 0 else 0

    # 全体平均（未知の顧客用）- 訓練データのみ
    default_monetary_stats = {}
    if customer_monetary_stats:
        all_stats_values = list(customer_monetary_stats.values())
        if all_stats_values:
            # 全ての顧客の統計量の特徴キーを取得
            all_keys = set()
            for stats in all_stats_values:
                all_keys.update(stats.keys())

            # 各特徴の全体平均を計算
            for key in all_keys:
                values = [stats.get(key, 0) for stats in all_stats_values]
                default_monetary_stats[key] = sum(values) / len(values) if values else 0

    # 全体平均（未知の顧客用）- 訓練+検証データ
    default_monetary_stats_with_valid = {}
    if customer_monetary_stats_with_valid:
        all_stats_values = list(customer_monetary_stats_with_valid.values())
        if all_stats_values:
            # 全ての顧客の統計量の特徴キーを取得
            all_keys = set()
            for stats in all_stats_values:
                all_keys.update(stats.keys())

            # 各特徴の全体平均を計算
            for key in all_keys:
                values = [stats.get(key, 0) for stats in all_stats_values]
                default_monetary_stats_with_valid[key] = sum(values) / len(values) if values else 0

    print(f"訓練データから計算された顧客単位の金銭特徴数: {len(customer_monetary_stats)}")
    print(f"訓練+検証データから計算された顧客単位の金銭特徴数: {len(customer_monetary_stats_with_valid)}")
    print(f"訓練データからのデフォルト金銭特徴数: {len(default_monetary_stats)}")
    print(f"訓練+検証データからのデフォルト金銭特徴数: {len(default_monetary_stats_with_valid)}")

    # ========================================
    # JANコードからカテゴリ情報を抽出
    # ========================================
    print("JANコードからカテゴリ情報を抽出しています...")

    # janデータフレームのカラム確認
    print(f"jan.csv カラム: {jan.columns.tolist()}")

    # カテゴリカラムの特定
    category_columns = [col for col in ['ディビジョン', '部門', 'カテゴリ', 'カテゴリ名', 'サブカテゴリ', 'ブランド']
                        if col in jan.columns]

    if not category_columns:
        print("警告: カテゴリ情報が含まれるカラムが見つかりません。代替カラムを使用します。")
        potential_category_cols = [col for col in jan.columns if col != 'JAN']
        category_columns = potential_category_cols[:min(5, len(potential_category_cols))]
        print(f"代替カテゴリカラム: {category_columns}")

    print(f"使用するカテゴリカラム: {category_columns}")

    # カテゴリの優先順位設定
    preferred_category_cols = ['カテゴリ', 'カテゴリ名', 'category', '部門', 'department', 'division', 'ディビジョン']

    # JANコードとカテゴリのマッピングを作成
    selected_category_col = None
    for col in preferred_category_cols:
        col_lower = col.lower()
        matching_cols = [c for c in category_columns if c.lower() == col_lower]
        if matching_cols:
            selected_category_col = matching_cols[0]
            print(f"カテゴリ情報として '{selected_category_col}' を使用します")
            break

    if not selected_category_col and category_columns:
        selected_category_col = category_columns[0]
        print(f"警告: 適切なカテゴリカラムが見つかりません。'{selected_category_col}'を使用します")

    # JANコードからカテゴリへのマッピングを作成
    print("JANコードからカテゴリへのマッピングを作成しています...")
    jan_to_category = {}

    if selected_category_col:
        # 選択したカラムの欠損値を確認
        jan['valid_category'] = jan[selected_category_col].fillna('')

        # 欠損値がある場合、他のカテゴリカラムから補完
        for col in category_columns:
            if col != selected_category_col:
                # 現在のカテゴリが空の行に対してのみ、他のカラムの値を使用
                jan.loc[jan['valid_category'] == '', 'valid_category'] = jan.loc[jan['valid_category'] == '', col].fillna('')

        # それでも値がない場合はJANコード自体を使用
        jan.loc[jan['valid_category'] == '', 'valid_category'] = jan.loc[jan['valid_category'] == '', 'JAN']

        # マッピング辞書を作成
        jan_to_category = dict(zip(jan['JAN'], jan['valid_category']))
    else:
        # カテゴリカラムがない場合はJANコードをそのまま使用
        jan_to_category = {row['JAN']: row['JAN'] for _, row in jan.iterrows()}

    # 売上数量カラムの特定
    quantity_cols = ['売上数量', '数量', '点数', '個数', 'quantity']
    quantity_col = next((col for col in quantity_cols if col in train_log.columns), None)

    if quantity_col:
        print(f"売上数量カラムとして '{quantity_col}' を使用します")
    else:
        print("警告: 売上数量カラムが見つかりません。'quantity'カラムを作成し1で埋めます")
        train_log['quantity'] = 1
        quantity_col = 'quantity'

    # JANコードをカテゴリに変換
    print("JANコードをカテゴリに変換しています...")
    # マッピング関数
    def get_category(jan_code):
        return jan_to_category.get(jan_code, "unknown")

    # ベクトル化されたマッピング適用
    train_log['mapped_category'] = train_log['JAN'].map(get_category)

    # 集計処理（訓練データのみで実行）
    print("セッションごとのカテゴリ集計を行っています...")

    # 不明カテゴリを除外
    filtered_log = train_log[train_log['mapped_category'] != "unknown"].copy()

    # カテゴリの一般的特性を学習するために全データを使用
    filtered_log_all = filtered_log.copy()  # 全データのコピーを保持
    # グループ集計（全データ使用）
    category_counts = filtered_log_all.groupby('mapped_category').size().reset_index(name='count')
    category_counts = category_counts.sort_values('count', ascending=False)

    # カテゴリの総数
    total_categories = len(category_counts)
    print(f"カテゴリの総数: {total_categories}")

    # 上位カテゴリを選択（訓練データの分布に基づく）
    N_TOP_CATEGORIES = min(100, total_categories)
    top_categories = category_counts.head(N_TOP_CATEGORIES)
    top_category_names = top_categories['mapped_category'].tolist()

    if top_category_names:
        displayed_categories = top_category_names[:min(10, len(top_category_names))]
        remaining = len(top_category_names) - len(displayed_categories)
        print(f"選択された上位{N_TOP_CATEGORIES}カテゴリ: {displayed_categories}... 他{remaining}カテゴリ")
    else:
        print("警告: カテゴリが見つかりません。特徴量が限定されます。")

    print("セッションごとのカテゴリピボットテーブルを作成しています...")
    # セッションごとの特徴を作成する段階で訓練データのフィルタリングを行う
    session_specific_log_train = filtered_log[filtered_log['session_id'].isin(train_session_ids)].copy()

    # カテゴリ特性には全データを使いつつ、セッション特徴は訓練データのみに制限
    train_session_category_pivot = session_specific_log_train.pivot_table(
        index='session_id',
        columns='mapped_category',
        values=quantity_col,
        aggfunc='sum',
        fill_value=0
    )

    # 上位カテゴリのみを選択
    if top_category_names:
        available_categories = [cat for cat in top_category_names if cat in train_session_category_pivot.columns]
        train_session_category_pivot = train_session_category_pivot[available_categories]

    # ========================================
    # セッション特徴量の作成
    # ========================================
    print("セッション特徴量を作成しています...")
    session_features = {}

    # 1. 訓練データの特徴量（ターゲット情報含む）
    for idx, row in tqdm(train_data.iterrows(), total=len(train_data), desc="訓練データの特徴量"):
        session_id = row['session_id']
        customer_id = row['顧客CD']

        # セッション金銭特徴を取得
        session_money_features = train_session_monetary_stats.get(session_id, {})
        # 顧客金銭特徴を取得
        customer_money_features = customer_monetary_stats.get(customer_id, default_monetary_stats)

        # 存在フラグ
        has_session_monetary = len(session_money_features) > 0
        has_customer_monetary = len(customer_money_features) > 0

        # 重要: ターゲットカテゴリの購入情報は最終的な特徴量から除外
        session_features[session_id] = {
            'hour': row['hour'],
            'dayofweek': row['dayofweek'],
            'day': row['day'],
            'month': row['month'],
            'time_period': row['time_period'],
            'is_holiday': row['is_holiday'],  # 祝日フラグを追加
            '店舗名_encoded': row['店舗名_encoded'],
            '年代_encoded': row['年代_encoded'],
            '性別_encoded': row['性別_encoded'],
            'is_train': True,
            'customer_id': customer_id,
            'target': {category: row[f'{category}_flag'] for category in CFG.CATEGORIES if f'{category}_flag' in row},
            # 金銭特徴を別々に追加
            'session_monetary_features': session_money_features,  # セッション固有
            'customer_monetary_features': customer_money_features,  # 顧客固有
            'has_session_monetary': has_session_monetary,  # 存在フラグ
            'has_customer_monetary': has_customer_monetary,  # 存在フラグ
        }

    # 2. 検証データの特徴量（ターゲット情報含む）
    for idx, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc="検証データの特徴量"):
        session_id = row['session_id']
        customer_id = row['顧客CD']

        # セッション金銭特徴を取得
        session_money_features = valid_session_monetary_stats.get(session_id, {})
        # 顧客金銭特徴を取得
        customer_money_features = customer_monetary_stats.get(customer_id, default_monetary_stats)

        # 存在フラグ
        has_session_monetary = len(session_money_features) > 0
        has_customer_monetary = len(customer_money_features) > 0

        session_features[session_id] = {
            'hour': row['hour'],
            'dayofweek': row['dayofweek'],
            'day': row['day'],
            'month': row['month'],
            'time_period': row['time_period'],
            'is_holiday': row['is_holiday'],  # 祝日フラグを追加
            '店舗名_encoded': row['店舗名_encoded'],
            '年代_encoded': row['年代_encoded'],
            '性別_encoded': row['性別_encoded'],
            'is_train': False,
            'is_valid': True,
            'customer_id': customer_id,
            'target': {category: row[f'{category}_flag'] for category in CFG.CATEGORIES if f'{category}_flag' in row},
            # 金銭特徴を別々に追加
            'session_monetary_features': session_money_features,  # セッション固有
            'customer_monetary_features': customer_money_features,  # 顧客固有
            'has_session_monetary': has_session_monetary,  # 存在フラグ
            'has_customer_monetary': has_customer_monetary,  # 存在フラグ
        }

    # 3. テストデータの特徴量（ターゲット情報なし）- 修正: 訓練+検証データの顧客特徴を使用
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="テストデータの特徴量"):
        session_id = row['session_id']
        customer_id = row['顧客CD']
        
        # セッション金銭特徴を取得（テストセッションには基本的に存在しない）
        session_money_features = {}
        # 顧客金銭特徴を取得（訓練+検証データから）
        customer_money_features = customer_monetary_stats_with_valid.get(customer_id, default_monetary_stats_with_valid)

        # 存在フラグ
        has_session_monetary = len(session_money_features) > 0
        has_customer_monetary = len(customer_money_features) > 0

        session_features[session_id] = {
            'hour': row['hour'],
            'dayofweek': row['dayofweek'],
            'day': row['day'],
            'month': row['month'],
            'time_period': row['time_period'],
            'is_holiday': row['is_holiday'],  # 祝日フラグを追加
            '店舗名_encoded': row['店舗名_encoded'],
            '年代_encoded': row['年代_encoded'],
            '性別_encoded': row['性別_encoded'],
            'is_train': False,
            'is_valid': False,
            'is_test': True,
            'customer_id': customer_id,
            # 金銭特徴を別々に追加
            'session_monetary_features': session_money_features,  # セッション固有
            'customer_monetary_features': customer_money_features,  # 顧客固有（訓練+検証データから）
            'has_session_monetary': has_session_monetary,  # 存在フラグ
            'has_customer_monetary': has_customer_monetary,  # 存在フラグ
        }

    # ターゲットカテゴリも含める
    exclude_categories = []  # ターゲットカテゴリも特徴として含める
    feature_categories = [c for c in top_category_names if c not in exclude_categories]

    # 訓練セッションIDのリスト
    train_session_ids_list = list(train_session_ids)

    # 結果を保存するための配列を初期化
    num_other_categories = len(train_session_category_pivot.columns)
    total_features = num_other_categories

    print(f"特徴量から除外するターゲットカテゴリ: {exclude_categories}")
    print(f"使用する特徴カテゴリ数: {num_other_categories}")
    print(f"合計特徴次元数: {total_features}")

    # ========================================
    # SVD次元圧縮
    # ========================================
    print("訓練セッションごとのカテゴリマトリックスを構築中...")

    # セッションIDを訓練・検証・テスト別に整理
    train_session_id_to_idx = {sid: i for i, sid in enumerate(train_session_ids_list)}
    valid_session_ids_list = list(valid_session_ids)
    test_session_ids_list = list(test_session_ids)

    # 訓練セッションのカテゴリマトリックスを構築
    train_category_matrix = np.zeros((len(train_session_ids_list), num_other_categories))

    # セッションとピボットテーブルのインデックスのマッピングを作成
    session_to_pivot_idx = {sid: i for i, sid in enumerate(train_session_category_pivot.index)}

    # 訓練セッションのカテゴリマトリックスに値を設定
    for i, session_id in enumerate(train_session_ids_list):
        if session_id in session_to_pivot_idx:
            pivot_idx = session_to_pivot_idx[session_id]
            if pivot_idx < len(train_session_category_pivot):
                train_category_matrix[i] = train_session_category_pivot.iloc[pivot_idx].values

    print(f"訓練セッションカテゴリマトリックスの形状: {train_category_matrix.shape}")

    # SVD次元圧縮（訓練データのみで学習）
    print("訓練データのセッションカテゴリマトリックスでSVD次元圧縮を実行しています...")
    num_features = train_category_matrix.shape[1]
    embed_dim = min(num_features, CFG.EMBED_DIM)

    if embed_dim < CFG.EMBED_DIM:
        print(f"警告: SVDの次元数を{CFG.EMBED_DIM}から{embed_dim}に縮小します (特徴量数: {num_features})")

    # SVD実行（訓練データのみでfit）
    if embed_dim < num_features:
        svd = TruncatedSVD(n_components=embed_dim, random_state=CFG.SEED)
        train_reduced_categories = svd.fit_transform(train_category_matrix)
        print(f"SVDを使用: {num_features}次元 → {embed_dim}次元")
        print(f"SVDの説明分散比: {np.sum(svd.explained_variance_ratio_):.4f}")
    else:
        # 次元削減の必要がなければそのまま使用
        train_reduced_categories = train_category_matrix
        svd = None
        print(f"SVDなしで使用: {num_features}次元")

    # 訓練セッションの圧縮特徴量をセッション特徴量に追加
    print("訓練セッションに圧縮特徴量を追加しています...")
    for i, session_id in enumerate(train_session_ids_list):
        if session_id in session_features:
            session_features[session_id]['category_embed'] = train_reduced_categories[i].tolist()

    # 検証・テストセッションのカテゴリ情報を取得
    print("検証・テストセッションのカテゴリ情報を収集しています...")

    # 検証データ用のログ取得
    if valid_session_ids:
        print("検証データのセッションログを処理しています...")
        # 検証セッションに対応するログを抽出
        valid_filtered_log = filtered_log[filtered_log['session_id'].isin(valid_session_ids)].copy()

        if len(valid_filtered_log) > 0:
            # 検証セッションのカテゴリピボットテーブル作成
            valid_session_category_pivot = valid_filtered_log.pivot_table(
                index='session_id',
                columns='mapped_category',
                values=quantity_col,
                aggfunc='sum',
                fill_value=0
            )

            # 訓練データと同じカテゴリセットを使用するために列を調整
            for category in top_category_names:
                if category not in valid_session_category_pivot.columns:
                    valid_session_category_pivot[category] = 0

            # 訓練データと同じカテゴリ順序を維持
            valid_session_category_pivot = valid_session_category_pivot[train_session_category_pivot.columns]

            # SVD変換を適用
            valid_category_matrix = valid_session_category_pivot.values
            if svd is not None:
                valid_reduced_categories = svd.transform(valid_category_matrix)
            else:
                valid_reduced_categories = valid_category_matrix

            # 検証セッションの特徴量に追加
            for i, session_id in enumerate(valid_session_category_pivot.index):
                if session_id in session_features:
                    session_features[session_id]['category_embed'] = valid_reduced_categories[i].tolist()

        # 特徴量がない検証セッションには0ベクトルを設定
        for session_id in valid_session_ids:
            if session_id in session_features and 'category_embed' not in session_features[session_id]:
                session_features[session_id]['category_embed'] = [0.0] * embed_dim

    # テストデータ用のログ取得（テストセッション自体の特性を活用）
    if test_session_ids:
        print("テストデータのセッションログを処理しています...")
        # 商品ログからテストセッションに関連するデータを抽出
        # 過去データに加えてテストセッション自体のカテゴリ情報を活用
        test_filtered_log = filtered_log_all[filtered_log_all['session_id'].isin(test_session_ids)].copy()

        if len(test_filtered_log) > 0:
            # テストセッションのカテゴリピボットテーブル作成
            test_session_category_pivot = test_filtered_log.pivot_table(
                index='session_id',
                columns='mapped_category',
                values=quantity_col,
                aggfunc='sum',
                fill_value=0
            )

            # 訓練データと同じカテゴリセットを使用するために列を調整
            adjusted_columns = []
            for category in train_session_category_pivot.columns:
                if category not in test_session_category_pivot.columns:
                    test_session_category_pivot[category] = 0
                adjusted_columns.append(category)

            # 訓練データと同じカテゴリ順序を維持
            test_session_category_pivot = test_session_category_pivot[adjusted_columns]

            # SVD変換を適用
            test_category_matrix = np.zeros((len(test_session_ids), len(adjusted_columns)))

            # セッションとピボットテーブルのインデックスのマッピングを作成
            test_session_to_pivot_idx = {sid: i for i, sid in enumerate(test_session_category_pivot.index)}

            # テストセッションのカテゴリマトリックスにデータを設定
            test_indices = []
            for i, session_id in enumerate(test_session_ids_list):
                if session_id in test_session_to_pivot_idx:
                    pivot_idx = test_session_to_pivot_idx[session_id]
                    if pivot_idx < len(test_session_category_pivot):
                        test_category_matrix[i] = test_session_category_pivot.iloc[pivot_idx].values
                        test_indices.append(i)

            # SVD変換を適用
            if svd is not None and len(test_indices) > 0:
                # 値がある行だけを変換
                test_matrix_filtered = test_category_matrix[test_indices]
                test_reduced_categories = svd.transform(test_matrix_filtered)

                # 結果を元の行に戻す
                for idx, orig_idx in enumerate(test_indices):
                    if test_session_ids_list[orig_idx] in session_features:
                        session_features[test_session_ids_list[orig_idx]]['category_embed'] = test_reduced_categories[idx].tolist()
            else:
                # SVDなしの場合
                for i, session_id in enumerate(test_session_ids_list):
                    if i in test_indices and session_id in session_features:
                        session_features[session_id]['category_embed'] = test_category_matrix[i].tolist()

            print(f"テストセッションのカテゴリ特徴を生成: {len(test_indices)}/{len(test_session_ids_list)}")

        # 特徴量を生成できなかったテストセッションには0ベクトルを設定
        for session_id in test_session_ids:
            if session_id in session_features and 'category_embed' not in session_features[session_id]:
                session_features[session_id]['category_embed'] = [0.0] * embed_dim

    # 実際に使用するEMBED_DIMを更新
    actual_embed_dim = embed_dim

    # ========================================
    # セッション系列の構築
    # ========================================
    print("セッション系列を構築しています...")
    X_train = []
    y_train = []
    X_valid = []
    y_valid = []

    # セッションIDとインデックスのマッピングを記録
    train_session_id_to_index = {}

    # 金銭特徴の正規化に使用する最大値を計算
    session_monetary_keys = set()
    customer_monetary_keys = set()

    for features in session_features.values():
        if 'session_monetary_features' in features:
            session_monetary_keys.update(features['session_monetary_features'].keys())
        if 'customer_monetary_features' in features:
            customer_monetary_keys.update(features['customer_monetary_features'].keys())

    session_monetary_max_values = {}
    customer_monetary_max_values = {}

    # それぞれの特徴タイプごとに最大値を計算
    for key in session_monetary_keys:
        values = [features.get('session_monetary_features', {}).get(key, 0) for features in session_features.values()]
        session_monetary_max_values[key] = max(values) if values else 1.0

    for key in customer_monetary_keys:
        values = [features.get('customer_monetary_features', {}).get(key, 0) for features in session_features.values()]
        customer_monetary_max_values[key] = max(values) if values else 1.0

    # 安全のため0除算を防止
    for key, value in session_monetary_max_values.items():
        if value == 0:
            session_monetary_max_values[key] = 1.0

    for key, value in customer_monetary_max_values.items():
        if value == 0:
            customer_monetary_max_values[key] = 1.0

    print(f"セッション金銭特徴のカラム数: {len(session_monetary_keys)}")
    print(f"顧客金銭特徴のカラム数: {len(customer_monetary_keys)}")

    # 正規化に使用する最大値を事前計算
    max_store_encoded = max(train_data['店舗名_encoded'].max(),
                           valid_data['店舗名_encoded'].max(),
                           test_data['店舗名_encoded'].max())

    max_age_encoded = max(train_data['年代_encoded'].max(),
                         valid_data['年代_encoded'].max(),
                         test_data['年代_encoded'].max())

    max_gender_encoded = max(train_data['性別_encoded'].max(),
                            valid_data['性別_encoded'].max(),
                            test_data['性別_encoded'].max())

    # 訓練データのセッションID一覧
    train_data_session_ids = []

    # 1. 訓練データのセッション系列構築
    print("訓練データのセッション系列を構築しています...")
    for customer_id, sessions in tqdm(train_customer_sessions.items(), desc="訓練セッション系列"):
        if len(sessions) < num_sessions:
            continue

        for i in range(len(sessions) - num_sessions + 1):
            # データ拡張のためにランダムにカット（指定確率）
            if random.random() < CFG.RANDOM_CUT_PROB and i > 0:
                start_idx = random.randint(i, len(sessions) - num_sessions)
            else:
                start_idx = i

            seq = sessions[start_idx:start_idx + num_sessions]
            target_session = seq[-1]

            # 入力系列の特徴量
            seq_features = []
            for session_id in seq:
                if session_id in session_features:
                    features = session_features[session_id]
                    # 基本特徴のみ使用
                    base_features = [
                        features['hour'] / 24.0,  # 正規化
                        features['dayofweek'] / 6.0,
                        features['day'] / 31.0,
                        features['month'] / 12.0,
                        features['time_period'] / 3.0,
                        features['is_holiday'],  # 祝日フラグ
                        features['店舗名_encoded'] / (max_store_encoded + 1),
                        features['年代_encoded'] / (max_age_encoded + 1),
                        features['性別_encoded'] / (max_gender_encoded + 1),
                    ]

                    # カテゴリ埋め込み
                    category_embed = features.get('category_embed', [0.0] * actual_embed_dim)

                    # 金銭特徴（セッション固有）
                    session_monetary_features = []
                    if 'session_monetary_features' in features:
                        for key in session_monetary_keys:
                            # 正規化して追加
                            value = features['session_monetary_features'].get(key, 0.0)
                            max_value = session_monetary_max_values.get(key, 1.0)
                            session_monetary_features.append(value / max_value)
                    else:
                        session_monetary_features = [0.0] * len(session_monetary_keys)

                    # 金銭特徴（顧客固有）
                    customer_monetary_features = []
                    if 'customer_monetary_features' in features:
                        for key in customer_monetary_keys:
                            # 正規化して追加
                            value = features['customer_monetary_features'].get(key, 0.0)
                            max_value = customer_monetary_max_values.get(key, 1.0)
                            customer_monetary_features.append(value / max_value)
                    else:
                        customer_monetary_features = [0.0] * len(customer_monetary_keys)

                    # 存在フラグ
                    has_session_monetary = [1.0 if features.get('has_session_monetary', False) else 0.0]
                    has_customer_monetary = [1.0 if features.get('has_customer_monetary', False) else 0.0]

                    # 全ての特徴を結合
                    combined_features = base_features + category_embed + session_monetary_features + customer_monetary_features + has_session_monetary + has_customer_monetary
                    seq_features.append(combined_features)
                else:
                    # 特徴がない場合はゼロパディング
                    seq_features.append([0] * (9 + actual_embed_dim + len(session_monetary_keys) + len(customer_monetary_keys) + 2))

            # ターゲットは最後のセッションのカテゴリ購入フラグ
            target = []
            if target_session in session_features:
                features = session_features[target_session]
                for category in CFG.CATEGORIES:
                    if 'target' in features and category in features['target']:
                        target.append(features['target'][category])
                    else:
                        target.append(0)
            else:
                target = [0] * len(CFG.CATEGORIES)

            # セッションIDとインデックスのマッピングを更新
            current_index = len(X_train)
            train_session_id_to_index[target_session] = current_index
            train_data_session_ids.append(target_session)

            X_train.append(seq_features)
            y_train.append(target)

    # 2. 検証データのセッション系列構築（訓練期間データと検証データを組み合わせ）
    print("検証データのセッション系列を構築しています...")
    valid_session_id_to_index = {}
    valid_data_session_ids = []

    for customer_id, valid_sessions in tqdm(valid_customer_sessions.items(), desc="検証セッション系列"):
        # この顧客の訓練期間のセッション（データリーク防止）
        train_sessions = train_customer_sessions.get(customer_id, [])

        for valid_session_id in valid_sessions:
            # 訓練期間の最後のセッションから順に取得（最新のものから最大num_sessions-1個）
            history_sessions = train_sessions[-min(num_sessions-1, len(train_sessions)):] if train_sessions else []

            # 履歴 + 現在の検証セッション
            complete_seq = history_sessions + [valid_session_id]

            # 合計がnum_sessions未満の場合はパディング
            if len(complete_seq) < num_sessions:
                complete_seq = ['padding'] * (num_sessions - len(complete_seq)) + complete_seq

            # 最新のnum_sessionsセッションのみ使用
            seq = complete_seq[-num_sessions:]

            # 入力系列の特徴量
            seq_features = []
            for session_id in seq:
                if session_id in session_features:
                    features = session_features[session_id]
                    # 基本特徴
                    base_features = [
                        features['hour'] / 24.0,
                        features['dayofweek'] / 6.0,
                        features['day'] / 31.0,
                        features['month'] / 12.0,
                        features['time_period'] / 3.0,
                        features['is_holiday'],  # 祝日フラグ
                        features['店舗名_encoded'] / (max_store_encoded + 1),
                        features['年代_encoded'] / (max_age_encoded + 1),
                        features['性別_encoded'] / (max_gender_encoded + 1),
                    ]

                    # カテゴリ埋め込み
                    category_embed = features.get('category_embed', [0.0] * actual_embed_dim)

                    # 金銭特徴（セッション固有）
                    session_monetary_features = []
                    if 'session_monetary_features' in features:
                        for key in session_monetary_keys:
                            # 正規化して追加
                            value = features['session_monetary_features'].get(key, 0.0)
                            max_value = session_monetary_max_values.get(key, 1.0)
                            session_monetary_features.append(value / max_value)
                    else:
                        session_monetary_features = [0.0] * len(session_monetary_keys)

                    # 金銭特徴（顧客固有）
                    customer_monetary_features = []
                    if 'customer_monetary_features' in features:
                        for key in customer_monetary_keys:
                            # 正規化して追加
                            value = features['customer_monetary_features'].get(key, 0.0)
                            max_value = customer_monetary_max_values.get(key, 1.0)
                            customer_monetary_features.append(value / max_value)
                    else:
                        customer_monetary_features = [0.0] * len(customer_monetary_keys)

                    # 存在フラグ
                    has_session_monetary = [1.0 if features.get('has_session_monetary', False) else 0.0]
                    has_customer_monetary = [1.0 if features.get('has_customer_monetary', False) else 0.0]

                    # 全ての特徴を結合
                    combined_features = base_features + category_embed + session_monetary_features + customer_monetary_features + has_session_monetary + has_customer_monetary
                    seq_features.append(combined_features)
                else:
                    # パディングまたは特徴がない場合はゼロ埋め
                    seq_features.append([0] * (9 + actual_embed_dim + len(session_monetary_keys) + len(customer_monetary_keys) + 2))

            # ターゲット
            target = []
            if valid_session_id in session_features:
                features = session_features[valid_session_id]
                for category in CFG.CATEGORIES:
                    if 'target' in features and category in features['target']:
                        target.append(features['target'][category])
                    else:
                        target.append(0)
            else:
                target = [0] * len(CFG.CATEGORIES)

            # インデックス更新
            current_index = len(X_valid)
            valid_session_id_to_index[valid_session_id] = current_index
            valid_data_session_ids.append(valid_session_id)

            X_valid.append(seq_features)
            y_valid.append(target)

    # 3. テストデータの特徴量作成（訓練+検証期間の履歴を活用）
    print("テストデータの特徴を作成しています...")
    X_test = []
    test_session_ids_list = []

    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="テストセッション系列"):
        customer_id = row['顧客CD']
        test_session_id = row['session_id']
        test_session_ids_list.append(test_session_id)

        # この顧客の訓練期間のセッションのみを使用（データリーク防止）
        history_sessions = train_customer_sessions.get(customer_id, [])

        # 検証期間のセッションも追加して履歴をより豊かにする
        valid_sessions = valid_customer_sessions.get(customer_id, [])
        history_sessions = history_sessions + valid_sessions

        # 訓練+検証期間の最後のセッションから順に取得（最新のものから最大num_sessions-1個）
        recent_history = history_sessions[-min(num_sessions-1, len(history_sessions)):] if history_sessions else []

        # テストセッションの特徴量
        features = session_features.get(test_session_id, {})

        # 基本特徴
        test_base_features = [
            row['hour'] / 24.0,
            row['dayofweek'] / 6.0,
            row['day'] / 31.0,
            row['month'] / 12.0,
            row['time_period'] / 3.0,
            row['is_holiday'],  # 祝日フラグ
            row['店舗名_encoded'] / (max_store_encoded + 1),
            row['年代_encoded'] / (max_age_encoded + 1),
            row['性別_encoded'] / (max_gender_encoded + 1),
        ]

        # カテゴリ埋め込み
        category_embed = features.get('category_embed', [0.0] * actual_embed_dim)

        # 金銭特徴（セッション固有）
        test_session_monetary_features = []
        if 'session_monetary_features' in features:
            for key in session_monetary_keys:
                # 正規化して追加
                value = features['session_monetary_features'].get(key, 0.0)
                max_value = session_monetary_max_values.get(key, 1.0)
                test_session_monetary_features.append(value / max_value)
        else:
            test_session_monetary_features = [0.0] * len(session_monetary_keys)

        # 金銭特徴（顧客固有）
        test_customer_monetary_features = []
        if 'customer_monetary_features' in features:
            for key in customer_monetary_keys:
                # 正規化して追加
                value = features['customer_monetary_features'].get(key, 0.0)
                max_value = customer_monetary_max_values.get(key, 1.0)
                test_customer_monetary_features.append(value / max_value)
        else:
            test_customer_monetary_features = [0.0] * len(customer_monetary_keys)

        # 存在フラグ
        test_has_session_monetary = [1.0 if features.get('has_session_monetary', False) else 0.0]
        test_has_customer_monetary = [1.0 if features.get('has_customer_monetary', False) else 0.0]

        # 全ての特徴を結合
        test_features = test_base_features + category_embed + test_session_monetary_features + test_customer_monetary_features + test_has_session_monetary + test_has_customer_monetary

        # 過去のセッション特徴量
        seq_features = []
        for session_id in recent_history:
            if session_id in session_features:
                features = session_features[session_id]
                # 基本特徴
                base_features = [
                    features['hour'] / 24.0,
                    features['dayofweek'] / 6.0,
                    features['day'] / 31.0,
                    features['month'] / 12.0,
                    features['time_period'] / 3.0,
                    features['is_holiday'],  # 祝日フラグ
                    features['店舗名_encoded'] / (max_store_encoded + 1),
                    features['年代_encoded'] / (max_age_encoded + 1),
                    features['性別_encoded'] / (max_gender_encoded + 1),
                ]

                # カテゴリ埋め込み
                category_embed = features.get('category_embed', [0.0] * actual_embed_dim)

                # 金銭特徴（セッション固有）
                session_monetary_features = []
                if 'session_monetary_features' in features:
                    for key in session_monetary_keys:
                        # 正規化して追加
                        value = features['session_monetary_features'].get(key, 0.0)
                        max_value = session_monetary_max_values.get(key, 1.0)
                        session_monetary_features.append(value / max_value)
                else:
                    session_monetary_features = [0.0] * len(session_monetary_keys)

                # 金銭特徴（顧客固有）
                customer_monetary_features = []
                if 'customer_monetary_features' in features:
                    for key in customer_monetary_keys:
                        # 正規化して追加
                        value = features['customer_monetary_features'].get(key, 0.0)
                        max_value = customer_monetary_max_values.get(key, 1.0)
                        customer_monetary_features.append(value / max_value)
                else:
                    customer_monetary_features = [0.0] * len(customer_monetary_keys)

                # 存在フラグ
                has_session_monetary = [1.0 if features.get('has_session_monetary', False) else 0.0]
                has_customer_monetary = [1.0 if features.get('has_customer_monetary', False) else 0.0]

                # 全ての特徴を結合
                combined_features = base_features + category_embed + session_monetary_features + customer_monetary_features + has_session_monetary + has_customer_monetary
                seq_features.append(combined_features)
            else:
                # 特徴がない場合はゼロパディング
                seq_features.append([0] * (9 + actual_embed_dim + len(session_monetary_keys) + len(customer_monetary_keys) + 2))

        # 過去のセッションが足りない場合はゼロパディング
        while len(seq_features) < num_sessions - 1:
            seq_features.insert(0, [0] * (9 + actual_embed_dim + len(session_monetary_keys) + len(customer_monetary_keys) + 2))

        # 最新のセッション（テストセッション）を最後に追加
        seq_features.append(test_features)

        X_test.append(seq_features)

    # 処理時間の計測
    elapsed_time = time.time() - start_time
    print(f"セッション系列作成の総処理時間: {elapsed_time:.2f}秒")

    # データ形状の確認
    print(f"訓練データ形状: X_train={np.array(X_train).shape}, y_train={np.array(y_train).shape}")
    print(f"検証データ形状: X_valid={np.array(X_valid).shape}, y_valid={np.array(y_valid).shape}")
    print(f"テストデータ形状: X_test={np.array(X_test).shape}")

    # セッションIDとインデックスのマッピング情報、および顧客ごとのセッション情報も返す
    return (np.array(X_train), np.array(y_train),
            np.array(X_valid), np.array(y_valid),
            np.array(X_test), test_session_ids_list,
            train_session_id_to_index, train_data_session_ids)


# ========================================
# PyTorch Dataset クラス
# ========================================
class SessionDataset(Dataset):
    """
    セッション系列データ用のPyTorchデータセット
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


# ========================================
# LSTM モデル定義
# ========================================
class SessionLSTM(nn.Module):
    """
    セッション系列を処理するLSTMモデル
    
    特徴:
    - 3つの並列LSTM層による特徴抽出
    - Dropoutによる正則化
    - マルチタスク学習（複数カテゴリの同時予測）
    """
    def __init__(self, input_dim, hidden_dim, num_categories, num_sessions):
        super(SessionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_sessions = num_sessions

        # Dropout層を追加（正則化のため）
        self.dropout = nn.Dropout(0.3)

        # 入力特徴量の前処理用NN
        self.feature_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 3つの並列LSTM層（アンサンブル効果を狙う）
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # 3つのLSTM出力を連結して最終予測を行う層
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            # Dropout層を出力層の前にも追加
            self.dropout,
            nn.Linear(hidden_dim, num_categories),
            nn.Sigmoid()  # 各カテゴリの購入確率を出力
        )

    def forward(self, x):
        """
        順伝播処理
        
        Args:
            x: 入力セッション系列 [batch_size, num_sessions, input_dim]
        
        Returns:
            各カテゴリの購入確率 [batch_size, num_categories]
        """
        batch_size = x.size(0)

        # 各時間ステップの入力を前処理
        x_nn = torch.zeros(batch_size, self.num_sessions, self.hidden_dim, device=x.device)
        for t in range(self.num_sessions):
            x_nn[:, t, :] = self.feature_nn(x[:, t, :])

        # Dropout層をLSTM入力の前に適用
        x_nn = self.dropout(x_nn)

        # 3つのLSTM層で並列処理
        lstm1_out, _ = self.lstm1(x_nn)
        lstm2_out, _ = self.lstm2(x_nn)
        lstm3_out, _ = self.lstm3(x_nn)

        # 最後の時間ステップの出力を取得
        lstm1_last = lstm1_out[:, -1, :]
        lstm2_last = lstm2_out[:, -1, :]
        lstm3_last = lstm3_out[:, -1, :]

        # 3つのLSTM出力を連結
        combined = torch.cat((lstm1_last, lstm2_last, lstm3_last), dim=1)

        # Dropout層をLSTM出力に適用
        combined = self.dropout(combined)

        # 最終予測
        output = self.output_layer(combined)

        return output


# ========================================
# 評価関数
# ========================================
def calculate_macro_auc(y_true, y_pred):
    """
    カテゴリごとのAUCを計算してMacro AUCを求める
    
    Args:
        y_true: 正解ラベル [samples, categories]
        y_pred: 予測確率 [samples, categories]
    
    Returns:
        tuple: (macro_auc, category_aucs)
    """
    aucs = []
    category_aucs = {}  # カテゴリごとのAUCを辞書で保存

    for i in range(CFG.NUM_CATEGORIES):
        # カテゴリごとにAUCを計算
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)
        category_aucs[i] = auc  # インデックスをキーとして保存
        print(f"Category {CFG.CATEGORIES[i]} AUC: {auc:.4f}")

    # Macro AUC（平均）を計算
    macro_auc = np.mean(aucs)
    return macro_auc, category_aucs


# ========================================
# カテゴリ別の重み付き損失関数
# ========================================
class WeightedCategoryBCELoss(nn.Module):
    """
    カテゴリごとに重み付けを行うBinary Cross Entropy Loss
    クラス不均衡に対応するため、少数クラスに高い重みを設定
    """
    def __init__(self, category_weights):
        super(WeightedCategoryBCELoss, self).__init__()
        self.category_weights = category_weights
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, outputs, targets):
        """
        重み付きBCE損失を計算
        
        Args:
            outputs: モデルの予測確率 [batch_size, num_categories]
            targets: 正解ラベル [batch_size, num_categories]
        
        Returns:
            重み付き損失値
        """
        # 各カテゴリに対する損失を計算
        loss = self.bce(outputs, targets)

        # カテゴリごとに重みを適用
        weighted_loss = 0
        for i in range(len(self.category_weights)):
            cat_loss = loss[:, i]
            # 正例には高い重み、負例には通常の重みを適用
            weights = torch.ones_like(cat_loss)
            weights[targets[:, i] > 0.5] = self.category_weights[i]
            weighted_loss += (cat_loss * weights).mean()

        return weighted_loss / len(self.category_weights)


def calculate_category_weights(y_train):
    """
    カテゴリごとのクラス重みを計算
    
    Args:
        y_train: 訓練ラベル [samples, categories]
    
    Returns:
        list: カテゴリごとの重み
    """
    weights = []
    for i in range(y_train.shape[1]):
        # i番目のカテゴリの正例と負例のカウント
        pos_count = np.sum(y_train[:, i] == 1)
        neg_count = np.sum(y_train[:, i] == 0)

        # 正例と負例の比率に基づいて重みを計算
        # 正例が少ない場合、その重みを高く設定
        ratio = neg_count / pos_count if pos_count > 0 else 1.0
        # 極端な重みを避けるために上限を設定
        ratio = min(ratio, 15.0)
        weights.append(ratio)
        print(f"Category {i} ({CFG.CATEGORIES[i]}): pos_count={pos_count}, neg_count={neg_count}, weight={ratio:.2f}")

    return weights


# ========================================
# モデル訓練関数
# ========================================
def train_model_fixed(X_train, y_train, X_valid, y_valid):
    """
    事前に時系列分割されたデータを使用してモデルをトレーニングする関数
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        X_valid: 検証データの特徴量
        y_valid: 検証データのラベル
    
    Returns:
        tuple: (trained_model, training_history)
    """
    # データセットとデータローダーの作成
    train_dataset = SessionDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)

    valid_dataset = SessionDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    print(f"Training on {len(X_train)} samples, validating on {len(X_valid)} samples")

    # 入力特徴量の次元数
    input_dim = X_train.shape[2]

    # デバイスの設定（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # モデルの初期化
    model = SessionLSTM(
        input_dim=input_dim,
        hidden_dim=CFG.HIDDEN_DIM,
        num_categories=CFG.NUM_CATEGORIES,
        num_sessions=CFG.NUM_SESSIONS
    ).to(device)

    # カテゴリごとの重みを計算
    category_weights = calculate_category_weights(y_train)

    # 損失関数とオプティマイザの設定
    # 重み付き損失関数を使用
    criterion = WeightedCategoryBCELoss(category_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=0.0001)

    # 早期停止のための変数
    best_valid_auc = 0
    best_epoch = 0
    patience = CFG.EARLY_STOPPING_PATIENCE
    counter = 0

    # 学習履歴
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_macro_auc': [],
        'category_aucs': []  # 各エポックのカテゴリごとのAUC
    }

    # 学習ループ
    for epoch in range(CFG.EPOCHS):
        # 訓練フェーズ
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Train]'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 勾配のリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = model(X_batch)

            # 損失の計算
            loss = criterion(outputs, y_batch)

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)

        # 検証フェーズ
        model.eval()
        total_valid_loss = 0
        all_valid_preds = []
        all_valid_targets = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(valid_dataloader, desc=f'Epoch {epoch+1}/{CFG.EPOCHS} [Valid]'):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 順伝播
                outputs = model(X_batch)

                # 損失の計算
                loss = criterion(outputs, y_batch)
                total_valid_loss += loss.item()

                # 予測と正解を保存
                all_valid_preds.append(outputs.cpu().numpy())
                all_valid_targets.append(y_batch.cpu().numpy())

        # 検証データ全体での損失とスコア
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        history['valid_loss'].append(avg_valid_loss)

        all_valid_preds = np.vstack(all_valid_preds)
        all_valid_targets = np.vstack(all_valid_targets)

        valid_macro_auc, category_aucs = calculate_macro_auc(all_valid_targets, all_valid_preds)
        history['valid_macro_auc'].append(valid_macro_auc)
        history['category_aucs'].append(category_aucs)

        print(f'Epoch {epoch+1}/{CFG.EPOCHS}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Valid Loss: {avg_valid_loss:.4f}')
        print(f'  Valid Macro AUC: {valid_macro_auc:.4f}')

        # 最良モデルの保存
        if valid_macro_auc > best_valid_auc:
            best_valid_auc = valid_macro_auc
            best_epoch = epoch
            counter = 0

            # モデルの保存
            os.makedirs(CFG.MODEL_PATH, exist_ok=True)
            torch.save(model.state_dict(), CFG.MODEL_PATH / 'best_lstm_model.pth')
            print(f'  New best model saved with Macro AUC: {valid_macro_auc:.4f}')
        else:
            counter += 1

        # 早期停止
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with Macro AUC: {best_valid_auc:.4f}')
            break

    # 最良モデルを読み込む
    model.load_state_dict(torch.load(CFG.MODEL_PATH / 'best_lstm_model.pth'))
    print(f'Loaded best model from epoch {best_epoch+1} with Macro AUC: {best_valid_auc:.4f}')

    return model, history


# ========================================
# 予測関数
# ========================================
def predict(model, X_test, test_session_ids, original_test_session_order):
    """
    テストデータに対する予測を行う関数
    訓練・検証データからの情報漏洩を防止
    
    Args:
        model: 訓練済みモデル
        X_test: テストデータの特徴量
        test_session_ids: テストセッションID
        original_test_session_order: 元のセッション順序
    
    Returns:
        予測結果のDataFrame
    """
    # GPUが利用可能ならGPUを使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"予測に使用するデバイス: {device}")

    # 評価モードに設定
    model.eval()

    # テストデータセットとデータローダーの作成
    test_dataset = SessionDataset(X_test)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    # 予測結果を格納するリスト
    predictions = []

    # 勾配計算を行わないように設定
    with torch.no_grad():
        for X_batch in tqdm(test_dataloader, desc='テストデータの予測'):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())

    # 予測結果を結合
    predictions = np.vstack(predictions)

    # カテゴリごとの予測分布を出力（デバッグ用）
    print("\n=== テスト予測の分布 ===")
    for i, category in enumerate(CFG.CATEGORIES):
        preds = predictions[:, i]
        print(f"{category} 予測分布:")
        print(f"  最小値: {preds.min():.4f}, 最大値: {preds.max():.4f}")
        print(f"  平均値: {preds.mean():.4f}, 中央値: {np.median(preds):.4f}")
        print(f"  標準偏差: {preds.std():.4f}")
        print(f"  >0.9: {(preds > 0.9).sum() / len(preds):.2%}, <0.1: {(preds < 0.1).sum() / len(preds):.2%}")

    # 異常な予測値がないか確認
    extreme_values = ((predictions > 0.99) | (predictions < 0.01)).sum()
    total_values = predictions.size
    print(f"極端な予測値の割合: {extreme_values}/{total_values} ({extreme_values/total_values:.2%})")

    # 予測結果をDataFrameに変換
    full_submission = pd.DataFrame({
        'session_id': test_session_ids,
        'チョコレート': predictions[:, 0],
        'ビール': predictions[:, 1],
        'ヘアケア': predictions[:, 2],
        '米（5㎏以下）': predictions[:, 3]
    })

    # 元のセッション順に予測結果を並べ替え
    print("元のセッション順に予測結果を並べ替えています...")
    full_submission = full_submission.set_index('session_id')
    full_submission = full_submission.reindex(index=original_test_session_order).reset_index()

    # session_idでユニークにする（重複がある場合は最後の値を使用）
    if len(full_submission) != len(set(test_session_ids)):
        print(f"警告: テストセッションIDに{len(full_submission) - len(set(test_session_ids))}個の重複があります。重複を削除します。")
        full_submission = full_submission.drop_duplicates(subset=['session_id'], keep='last')

    # テスト予測を保存（オプション）
    if CFG.SAVE_TEST_PREDS:
        # 予測値をそのまま保存（後処理や分析用）
        os.makedirs(CFG.OOF_DATA_PATH, exist_ok=True)
        raw_preds_df = pd.DataFrame(predictions, columns=CFG.CATEGORIES)
        raw_preds_df['session_id'] = test_session_ids
        raw_preds_df.to_csv(CFG.OOF_DATA_PATH / 'test_predictions_isolated.csv', index=False)
        print(f"生の予測値を保存しました: {CFG.OOF_DATA_PATH / 'test_predictions_isolated.csv'}")

    # 提出用ファイル（session_idなし）
    submission = full_submission[CFG.CATEGORIES].copy()
    print(f"最終提出データの形状: {submission.shape}")

    return submission


# ========================================
# 結果可視化関数
# ========================================
def plot_training_history(history):
    """
    訓練履歴をプロットして保存
    
    Args:
        history: 訓練履歴の辞書
    """
    try:
        import matplotlib.pyplot as plt

        # プロットの作成
        plt.figure(figsize=(15, 5))

        # 損失のプロット
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # AUCのプロット
        plt.subplot(1, 2, 2)
        plt.plot(history['valid_macro_auc'], label='Validation Macro AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Macro AUC')
        plt.title('Validation Macro AUC')
        plt.legend()

        plt.tight_layout()

        # グラフを表示
        plt.show()

        # 保存
        plt.savefig(CFG.OUTPUT_PATH / 'training_history.png')
        plt.close()

        print(f"Training history plot saved to {CFG.OUTPUT_PATH / 'training_history.png'}")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")


# ========================================
# メイン関数
# ========================================
def main():
    """
    メイン実行関数（完全データ分離版）
    """
    # シード固定
    seed_everything(CFG.SEED)

    # デバッグモードの表示
    if CFG.DEBUG_MODE:
        print("="*50)
        print(f"RUNNING IN DEBUG MODE WITH {CFG.DEBUG_SAMPLE_SIZE} SAMPLES")
        print("="*50)

    # 出力ディレクトリの作成
    os.makedirs(CFG.OUTPUT_PATH, exist_ok=True)
    os.makedirs(CFG.MODEL_PATH, exist_ok=True)
    os.makedirs(CFG.OOF_DATA_PATH, exist_ok=True)

    # データ読み込み
    print("完全分離データ読み込みを実行中...")
    (train_session_df, valid_session_df, test_session_df,
     train_log_df, train_target_df, valid_target_df, jan_df,
     original_test_session_order) = load_data()

    # 特徴量エンジニアリング（完全分離版）
    print("完全分離特徴量エンジニアリングを実行中...")
    train_data, valid_data, test_data, train_log, jan, encoders_info = feature_engineering_complete_isolation(
        train_session_df, valid_session_df, test_session_df,
        train_log_df, train_target_df, valid_target_df, jan_df
    )

    # セッション系列の作成（完全分離版）
    print("完全分離セッション系列作成を実行中...")
    (X_train, y_train, X_valid, y_valid, X_test,
     test_session_ids, train_session_id_to_index,
     train_data_session_ids) = create_session_sequences(
        train_data, valid_data, test_data, train_log, jan,
        encoders_info, num_sessions=CFG.NUM_SESSIONS
    )

    print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation data shape: X_valid={X_valid.shape}, y_valid={y_valid.shape}")
    print(f"Test data shape: X_test={X_test.shape}, test_session_count={len(test_session_ids)}")

    # デバッグモードでは少ないエポック数で実行
    original_epochs = CFG.EPOCHS
    if CFG.DEBUG_MODE:
        CFG.EPOCHS = min(5, CFG.EPOCHS)  # デバッグ時は最大5エポック
        print(f"Debug mode: Reducing epochs from {original_epochs} to {CFG.EPOCHS}")

    # モデルの学習（時系列検証付き）
    print("\nTraining model with fully isolated data...")
    model, history = train_model_fixed(X_train, y_train, X_valid, y_valid)

    # 学習履歴の可視化
    plot_training_history(history)

    # テストデータに対する予測
    print("\nMaking predictions on completely isolated test data...")
    submission = predict(model, X_test, test_session_ids, original_test_session_order)

    # 提出ファイルの保存
    prefix = 'debug_' if CFG.DEBUG_MODE else ''
    submission_path = CFG.OUTPUT_PATH / f"{prefix}lstm_submission_complete_isolation.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # 最終のMacro AUCスコアを表示
    best_valid_auc = max(history['valid_macro_auc'])
    print(f"\nBest validation Macro AUC: {best_valid_auc:.4f}")

    # デバッグモードを元に戻す
    if CFG.DEBUG_MODE:
        CFG.EPOCHS = original_epochs

    # 最後のvalidationで得られたカテゴリごとのAUCを取得
    last_category_aucs = {}
    if 'category_aucs' in history and history['category_aucs']:
        last_category_aucs = history['category_aucs'][-1]

    # 結果のサマリー表示
    print("\n=== Training Summary ===")
    for i, category in enumerate(CFG.CATEGORIES):
        if i in last_category_aucs:
            category_auc = last_category_aucs[i]
        else:
            # カテゴリAUCが取得できない場合は-1を表示
            category_auc = -1
        print(f"{category} AUC: {category_auc:.4f}")

    print(f"Final Macro AUC: {best_valid_auc:.4f}")
    print("=======================")

    # カテゴリごとの予測値分布を分析（データリーク診断のため）
    if len(X_valid) > 0:
        print("\n=== Category Prediction Analysis ===")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        valid_dataset = SessionDataset(X_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for X_batch in valid_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_preds.append(outputs.cpu().numpy())

        all_preds = np.vstack(all_preds)

        # カテゴリごとの予測値分布を分析
        for i, category in enumerate(CFG.CATEGORIES):
            preds = all_preds[:, i]
            print(f"{category} prediction stats:")
            print(f"  Min: {preds.min():.4f}, Max: {preds.max():.4f}")
            print(f"  Mean: {preds.mean():.4f}, Median: {np.median(preds):.4f}")
            print(f"  25th: {np.percentile(preds, 25):.4f}, 75th: {np.percentile(preds, 75):.4f}")
            print(f"  >0.9: {(preds > 0.9).sum() / len(preds):.2%}, <0.1: {(preds < 0.1).sum() / len(preds):.2%}")

        print("================================")

    return model, submission, history


# ========================================
# スクリプト実行部分
# ========================================
if __name__ == "__main__":
    model, submission, history = main()

    print("\nTraining completed with complete data isolation!")
    print(f"Check output directory for results: {CFG.OUTPUT_PATH}")

    # 提案事項と改善点の表示
    print("\n=== Possible Improvements ===")
    print("1. 特徴量の追加: 祝日フラグ、給料日フラグ、イベント情報などの追加")
    print("2. ハイパーパラメータの最適化: Optuna/Ray Tuneを使った自動最適化")
    print("3. モデルの拡張: Attention機構の追加、Transformer系モデルの検討")
    print("4. アンサンブル: LSTMと決定木系モデル(LightGBM)の組み合わせ")
    print("5. 時系列交差検証: 複数の時間窓を使った安定したCVスコアの獲得")
    print("6. 顧客セグメント特徴の活用: より抽象的なセグメント特徴の導入")
    print("============================")

        print(f"警告: 検証データとテストデータの間に{len(valid_test_overlap)}個のセッションIDの重複があります。")

    # 訓練セッションに対応するログデータを抽出（データリーク防止）
    train_log_pd = train_log_df.to_pandas()
    train_log_filtered = train_log_pd[train_log_pd['session_id'].isin(train_session_ids)].copy()
    train_log_df_filtered = pl.from_pandas(train_log_filtered)

    print(f"フィルタリング後の訓練ログデータ行数: {len(train_log_df_filtered)}")

    # ターゲットデータも同様にフィルタリング
    train_target_pd = train_target_df.to_pandas()

    # 訓練データのターゲット
    train_target_filtered = train_target_pd[train_target_pd['session_id'].isin(train_session_ids)].copy()
    train_target_df_filtered = pl.from_pandas(train_target_filtered)

    # 検証データのターゲット
    valid_target_filtered = train_target_pd[train_target_pd['session_id'].isin(valid_session_ids)].copy()
    valid_target_df_filtered = pl.from_pandas(valid_target_filtered)

    print(f"フィルタリング後の訓練ターゲットデータ行数: {len(train_target_df_filtered)}")
    print(f"フィルタリング後の検証ターゲットデータ行数: {len(valid_target_df_filtered)}")

    return (actual_train_session_df, valid_session_df, test_session_df,
            train_log_df_filtered, train_target_df_filtered, valid_target_df_filtered, jan_df,
            original_test_session_order)


# ========================================
# 特徴量エンジニアリング関数
# ========================================
def feature_engineering_complete_isolation(train_session_df, valid_session_df, test_session_df,
                                          train_log_df, train_target_df, valid_target_df, jan_df):
    """
    訓練データ、検証データ、テストデータを完全に分離して特徴量エンジニアリングを行う
    各変換器を訓練データのみで学習し、検証・テストデータには変換のみを適用
    
    Args:
        train_session_df: 訓練セッションデータ
        valid_session_df: 検証セッションデータ
        test_session_df: テストセッションデータ
        train_log_df: 訓練ログデータ
        train_target_df: 訓練ターゲットデータ
        valid_target_df: 検証ターゲットデータ
        jan_df: 商品マスタデータ
    
    Returns:
        tuple: 前処理済みのデータセット
    """
    # Polarからpandasに変換して処理
    train_session = train_session_df.to_pandas()
    valid_session = valid_session_df.to_pandas()
    test_session = test_session_df.to_pandas()

    train_log = train_log_df.to_pandas()
    train_target = train_target_df.to_pandas()
    valid_target = valid_target_df.to_pandas() if valid_target_df is not None else None
    jan = jan_df.to_pandas()

    print(f"訓練セッションデータ: {len(train_session)}行")
    print(f"検証セッションデータ: {len(valid_session)}行")
    print(f"テストセッションデータ: {len(test_session)}行")

    # 日付型に変換
    train_session['売上日'] = pd.to_datetime(train_session['売上日'])
    valid_session['売上日'] = pd.to_datetime(valid_session['売上日'])
    test_session['売上日'] = pd.to_datetime(test_session['売上日'])

    print(f"訓練データ期間: {train_session['売上日'].min()} 〜 {train_session['売上日'].max()}")
    print(f"検証データ期間: {valid_session['売上日'].min()} 〜 {valid_session['売上日'].max()}")
    print(f"テストデータ期間: {test_session['売上日'].min()} 〜 {test_session['売上日'].max()}")

    def basic_preprocess(session_data):
        """
        基本的な前処理（統計量に依存しない特徴量作成）
        
        Args:
            session_data: セッションデータ
        
        Returns:
            前処理済みセッションデータ
        """
        # 時刻列の処理（数値形式または時刻形式対応）
        try:
            # 時刻が「HH:MM:SS」形式の場合
            session_data['時刻'] = pd.to_datetime(session_data['時刻'], format='%H:%M:%S').dt.time
            session_data['hour'] = pd.to_datetime(session_data['時刻'], format='%H:%M:%S').dt.hour
        except ValueError:
            # 時刻が数値（時間のみ）の場合
            print("時刻列が数値形式です。時間として処理します。")
            session_data['hour'] = session_data['時刻'].astype(int)
            from datetime import time
            session_data['時刻'] = session_data['hour'].apply(lambda x: time(hour=int(x)))

        # 日付関連特徴量の作成
        session_data['dayofweek'] = session_data['売上日'].dt.dayofweek  # 曜日（0=月曜）
        session_data['day'] = session_data['売上日'].dt.day              # 日
        session_data['month'] = session_data['売上日'].dt.month          # 月

        # 時間帯特徴量（朝・昼・夕・夜）
        def get_time_period(hour):
            """時間帯を4つの区間に分類"""
            if 5 <= hour < 11:
                return 0  # 朝
            elif 11 <= hour < 15:
                return 1  # 昼
            elif 15 <= hour < 19:
                return 2  # 夕
            else:
                return 3  # 夜

        session_data['time_period'] = session_data['hour'].apply(get_time_period)

        return session_data

    # 基本前処理の実行
    print("訓練データの基本前処理を実行中...")
    train_data = basic_preprocess(train_session)

    print("検証データの基本前処理を実行中...")
    valid_data = basic_preprocess(valid_session)

    print("テストデータの基本前処理を実行中...")
    test_data = basic_preprocess(test_session)

    # カテゴリ型特徴量のエンコーディング（訓練データでfit、他は変換のみ）
    print("カテゴリ型特徴量のエンコーディング（訓練データのみで学習）...")
    encoders = {}

    # 店舗名のラベルエンコーディング
    encoders['le_store'] = LabelEncoder()
    train_data['店舗名_encoded'] = encoders['le_store'].fit_transform(train_data['店舗名'])

    # 年代のラベルエンコーディング
    encoders['le_age'] = LabelEncoder()
    train_data['年代_encoded'] = encoders['le_age'].fit_transform(train_data['年代'])

    # 性別のラベルエンコーディング
    encoders['le_gender'] = LabelEncoder()
    train_data['性別_encoded'] = encoders['le_gender'].fit_transform(train_data['性別'])

    def safe_transform(encoder, series, default_value=-1):
        """
        安全に変換を適用し、未知のカテゴリに対してはデフォルト値を使用
        
        Args:
            encoder: 学習済みLabelEncoder
            series: 変換対象のシリーズ
            default_value: 未知カテゴリに対するデフォルト値
        
        Returns:
            変換済みの値
        """
        result = np.ones(len(series), dtype=int) * default_value

        # 既知のカテゴリのみ変換
        for i, val in enumerate(series):
            try:
                result[i] = encoder.transform([val])[0]
            except ValueError:
                pass  # デフォルト値のまま

        return result

    # 検証データにエンコーダを適用
    print("検証データにエンコーダを適用中...")
    valid_data['店舗名_encoded'] = safe_transform(encoders['le_store'], valid_data['店舗名'])
    valid_data['年代_encoded'] = safe_transform(encoders['le_age'], valid_data['年代'])
    valid_data['性別_encoded'] = safe_transform(encoders['le_gender'], valid_data['性別'])

    # テストデータにエンコーダを適用
    print("テストデータにエンコーダを適用中...")
    test_data['店舗名_encoded'] = safe_transform(encoders['le_store'], test_data['店舗名'])
    test_data['年代_encoded'] = safe_transform(encoders['le_age'], test_data['年代'])
    test_data['性別_encoded'] = safe_transform(encoders['le_gender'], test_data['性別'])

    # ターゲット変数の結合
    print("ターゲット変数の結合...")

    # 訓練データとターゲットのマージ
    if 'session_id' in train_target.columns:
        train_data = pd.merge(train_data, train_target, on='session_id', how='left')
    else:
        raise ValueError("train_targetにsession_idが含まれていません。")

    # 検証データとターゲットのマージ
    if valid_target is not None and len(valid_target) > 0:
        if 'session_id' in valid_target.columns:
            valid_data = pd.merge(valid_data, valid_target, on='session_id', how='left')
        else:
            raise ValueError("valid_targetにsession_idが含まれていません。")

    # NaNをゼロで埋める
    for category in CFG.CATEGORIES:
        if category in train_data.columns:
            train_data[category] = train_data[category].fillna(0)
        if category in valid_data.columns:
            valid_data[category] = valid_data[category].fillna(0)

    # カテゴリ購入フラグの作成（0/1のバイナリ特徴）
    print("カテゴリ購入フラグを作成しています...")
    for category in CFG.CATEGORIES:
        if category in train_data.columns:
            train_data[f'{category}_flag'] = (train_data[category] > 0).astype(int)
        if category in valid_data.columns:
            valid_data[f'{category}_flag'] = (valid_data[category] > 0).astype(int)

    # データ形状の確認
    print(f"訓練データのカラム: {train_data.columns.tolist()}")
    print(f"訓練データ shape: {train_data.shape}")
    print(f"検証データ shape: {valid_data.shape}")
    print(f"テストデータ shape: {test_data.shape}")

    # 顧客情報の収集（データリーク防止のため、各セットで独立して収集）
    print("顧客情報を収集中...")

    train_customers = set(train_data['顧客CD'].unique())
    valid_customers = set(valid_data['顧客CD'].unique())
    test_customers = set(test_data['顧客CD'].unique())

    # 共通顧客の確認
    train_valid_common = train_customers.intersection(valid_customers)
    train_test_common = train_customers.intersection(test_customers)
    valid_test_common = valid_customers.intersection(test_customers)

    print(f"訓練・検証に共通の顧客数: {len(train_valid_common)}")
    print(f"訓練・テストに共通の顧客数: {len(train_test_common)}")
    print(f"検証・テストに共通の顧客数: {len(valid_test_common)}")

    # 完全にユニークな顧客数
    train_only = train_customers - valid_customers - test_customers
    valid_only = valid_customers - train_customers - test_customers
    test_only = test_customers - train_customers - valid_customers

    print(f"訓練データのみに存在する顧客数: {len(train_only)}")
    print(f"検証データのみに存在する顧客数: {len(valid_only)}")
    print(f"テストデータのみに存在する顧客数: {len(test_only)}")

    # エンコーダ情報をまとめて返す
    encoders_info = {
        'le_store': encoders['le_store'],
        'le_age': encoders['le_age'],
        'le_gender': encoders['le_gender'],
        'train_customers': train_customers,
        'valid_customers': valid_customers,
        'test_customers': test_customers
    }

    return train_data, valid_data, test_data, train_log, jan, encoders_info


# ========================================
# セッション系列作成関数（メイン関数）
# ========================================
def create_session_sequences(train_data, valid_data, test_data, train_log, jan, encoders_info, num_sessions=5):
    """
    訓練・検証・テストデータを完全に分離し、データリークを防ぐセッション系列作成関数
    セッションごとのカテゴリ情報を使用して特徴量を作成する改良版
    祝日フラグと顧客ベースの金銭関連特徴を追加
    
    Args:
        train_data: 訓練データ
        valid_data: 検証データ
        test_data: テストデータ
        train_log: 訓練ログデータ
        jan: 商品マスタデータ
        encoders_info: エンコーダ情報
        num_sessions: 使用するセッション数
    
    Returns:
        tuple: LSTMモデル用の系列データ
    """
    # 必要なライブラリをインポート
    import pandas as pd
    import numpy as np
    import polars as pl
    from scipy import sparse
    import time
    from collections import defaultdict
    from sklearn.decomposition import TruncatedSVD

    # パフォーマンス計測開始
    start_time = time.time()

    # デバッグモード対応（データ量制限）
    if CFG.DEBUG_MODE:
        print(f"DEBUG MODE: Using only {CFG.DEBUG_SAMPLE_SIZE} samples")
        # 各データセットのサンプル数を制限
        unique_customers = train_data['顧客CD'].unique()
        if len(unique_customers) > CFG.DEBUG_SAMPLE_SIZE:
            selected_customers = np.random.choice(unique_customers, CFG.DEBUG_SAMPLE_SIZE, replace=False)
            train_data = train_data[train_data['顧客CD'].isin(selected_customers)].reset_index(drop=True)

        if len(valid_data) > CFG.DEBUG_SAMPLE_SIZE:
            valid_data = valid_data.sample(CFG.DEBUG_SAMPLE_SIZE, random_state=CFG.SEED).reset_index(drop=True)

        if len(test_data) > CFG.DEBUG_SAMPLE_SIZE:
            test_data = test_data.sample(CFG.DEBUG_SAMPLE_SIZE, random_state=CFG.SEED).reset_index(drop=True)

        print(f"Debug train data shape: {train_data.shape}")
        print(f"Debug valid data shape: {valid_data.shape}")
        print(f"Debug test data shape: {test_data.shape}")

    # 祝日カレンダーの作成
    def create_holiday_calendar():
        """2024年の日本の祝日を手動で設定する関数"""
        holidays_2024 = [
            "2024-01-01",  # 元日
            "2024-01-08",  # 成人の日
            "2024-02-11",  # 建国記念の日
            "2024-02-12",  # 振替休日
            "2024-02-23",  # 天皇誕生日
            "2024-03-20",  # 春分の日
            "2024-04-29",  # 昭和の日
            "2024-05-03",  # 憲法記念日
            "2024-05-04",  # みどりの日
            "2024-05-05",  # こどもの日
            "2024-05-06",  # 振替休日
            "2024-07-15",  # 海の日
            "2024-08-11",  # 山の日
            "2024-08-12",  # 振替休日
            "2024-09-16",  # 敬老の日
            "2024-09-22",  # 秋分の日
            "2024-09-23",  # 振替休日
            "2024-10-14",  # スポーツの日
            "2024-11-03",  # 文化の日
            "2024-11-04",  # 振替休日
            "2024-11-23",  # 勤労感謝の日
        ]
        return set(holidays_2024)

    # 祝日カレンダーの作成
    HOLIDAY_SET = create_holiday_calendar()

    def is_holiday(date):
        """日付が祝日または土日かどうかを判定する関数"""
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
            # 祝日リストに含まれるかチェック
            if date_str in HOLIDAY_SET:
                return 1
            # 土日判定
            if date.dayofweek >= 5:  # 5=土曜日, 6=日曜日
                return 1
        return 0

    # 各データセットを時系列で並べる
    print("各データセットを時系列で並べています...")
    train_data = train_data.sort_values(['顧客CD', '売上日', '時刻']).reset_index(drop=True)
    valid_data = valid_data.sort_values(['顧客CD', '売上日', '時刻']).reset_index(drop=True)
    test_data = test_data.sort_values(['顧客CD', '売上日', '時刻']).reset_index(drop=True)

    # 祝日フラグの追加
    print("祝日フラグを追加しています...")
    train_data['is_holiday'] = train_data['売上日'].apply(is_holiday)
    valid_data['is_holiday'] = valid_data['売上日'].apply(is_holiday)
    test_data['is_holiday'] = test_data['売上日'].apply(is_holiday)

    # セッションIDセットの作成
    train_session_ids = set(train_data['session_id'])
    valid_session_ids = set(valid_data['session_id'])
    test_session_ids = set(test_data['session_id'])

    # セッションIDの重複確認（安全チェック）
    train_valid_overlap = train_session_ids.intersection(valid_session_ids)
    train_test_overlap = train_session_ids.intersection(test_session_ids)
    valid_test_overlap = valid_session_ids.intersection(test_session_ids)

    if train_valid_overlap:
        print(f"警告: 訓練・検証データ間に{len(train_valid_overlap)}個のセッションID重複があります。")
    if train_test_overlap:
        print(f"警告: 訓練・テストデータ間に{len(train_test_overlap)}個のセッションID重複があります。")
    if valid_test_overlap
