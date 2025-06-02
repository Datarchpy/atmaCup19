# =============================================================================
# ベイズ推定強化予測モデル v1.0
# 
# 機能概要:
# - 時間的特徴量の拡充（日付、時刻、祝日、月初月末等）
# - ベイズ推定による顧客購入確率の計算
# - カテゴリごとの最適パラメータ調整（Optuna使用）
# - カテゴリ購入シーケンスの特徴化
# - フィードバックループによる予測精度向上
# =============================================================================

import pandas as pd
import numpy as np
import datetime
import optuna
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# =========================================
# 基本設定・定数定義
# =========================================

# 予測対象カテゴリの定義
target_col_list = ['チョコレート', 'ビール', 'ヘアケア', '米（5㎏以下）']

# 日本の祝日リスト（2023-2024年）
# 購入パターンに影響する可能性がある祝日を特徴量として活用
JP_HOLIDAYS = [
    '2023-01-01', '2023-01-02', '2023-01-09', '2023-02-11', '2023-02-23', 
    '2023-03-21', '2023-04-29', '2023-05-03', '2023-05-04', '2023-05-05', 
    '2023-07-17', '2023-08-11', '2023-09-18', '2023-09-23', '2023-10-09', 
    '2023-11-03', '2023-11-23', '2023-12-23',
    '2024-01-01', '2024-01-08', '2024-02-11', '2024-02-12', '2024-02-23', 
    '2024-03-20', '2024-04-29', '2024-05-03', '2024-05-04', '2024-05-05', 
    '2024-05-06', '2024-07-15', '2024-08-11', '2024-08-12', '2024-09-16', 
    '2024-09-22', '2024-09-23', '2024-10-14', '2024-11-03', '2024-11-04', 
    '2024-11-23', '2024-12-23'
]
JP_HOLIDAYS = pd.to_datetime(JP_HOLIDAYS)

# =========================================
# データ読み込み関数
# =========================================

def load_data(data_path='/kaggle/input/atmacup19-dataset/atmacup19_dataset'):
    """
    必要なデータファイルを読み込む
    
    Args:
        data_path (str): データファイルのパス
    
    Returns:
        tuple: 各データフレーム（訓練セッション、テストセッション、ターゲット、ログ、JANコード）
    """
    # 各データファイルの読み込み
    train_session_df = pd.read_csv(f'{data_path}/train_session.csv')  # 訓練用セッションデータ
    test_session_df = pd.read_csv(f'{data_path}/test_session.csv')    # テスト用セッションデータ
    train_target_df = pd.read_csv(f'{data_path}/train_target.csv')    # 訓練用ターゲットデータ
    train_log_df = pd.read_csv(f'{data_path}/train_log.csv')          # 購入ログデータ
    jan_df = pd.read_csv(f'{data_path}/jan.csv')                      # JANコードマスタ
    
    # train_target_dfにsession_idカラムを追加（データ整合性確保のため）
    if 'session_id' not in train_target_df.columns:
        # train_session_dfと同じ順序であることを前提に、session_idを追加
        train_target_df['session_id'] = train_session_df['session_id'].values
    
    # データサイズの確認
    print(f"訓練セッションデータサイズ: {train_session_df.shape}")
    print(f"テストセッションデータサイズ: {test_session_df.shape}")
    print(f"訓練ターゲットデータサイズ: {train_target_df.shape}")
    print(f"購入ログデータサイズ: {train_log_df.shape}")
    print(f"JANコードデータサイズ: {jan_df.shape}")
    
    return train_session_df, test_session_df, train_target_df, train_log_df, jan_df

# =========================================
# 拡張特徴量エンジニアリング
# =========================================

def enhanced_preprocess_data(df, jan_df=None, train_log_df=None, train_targets=None):
    """
    拡張された特徴量エンジニアリング
    
    Args:
        df: 処理対象の元データ
        jan_df: JANコードマスタ（オプション）
        train_log_df: 商品購入ログデータ（オプション）
        train_targets: ターゲット情報（オプション）
    
    Returns:
        DataFrame: 拡張された特徴量を持つデータフレーム
    """
    df_copy = df.copy()
    
    # =================
    # バリデーションモードの処理
    # =================
    if validate:
        print("\n11. バリデーション評価とエラー分析を実行...")
        # スコア計算前にNaN値をチェック・修正
        nan_check = final_pred.isna().sum().sum()
        if nan_check > 0:
            print(f"警告: スコア計算前に {nan_check} 個のNaN値を修正します。")
            final_pred = final_pred.fillna(0)
        
        # バリデーション予測値をCSVファイルに保存
        validation_pred_filename = f'bayes_validation_pred_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        final_pred.to_csv(validation_pred_filename, index=False)
        print(f"バリデーション予測を {validation_pred_filename} として保存しました。")
        
        # 基本スコア計算
        try:
            basic_score = calc_score(validation_true, final_pred, target_col_list)
            print(f"基本予測スコア: {basic_score:.6f}")
            
            # カテゴリ別スコアの計算
            for cat in target_col_list:
                cat_score = roc_auc_score(validation_true[cat], final_pred[cat])
                print(f"  {cat} スコア: {cat_score:.6f}")
                
        except Exception as e:
            print(f"スコア計算でエラーが発生しました: {e}")
            basic_score = 0
        
        # エラーパターンの分析
        analyze_error_patterns(
            validation_df, 
            final_pred, 
            validation_true, 
            target_col_list
        )
        
        # エラー修正モデルの構築
        print("\n12. エラー修正モデルを構築中...")
        error_models = create_error_correction_model(
            validation_df, 
            final_pred, 
            validation_true, 
            target_col_list
        )
        
        # エラー修正の適用
        print("13. エラー修正を適用中...")
        corrected_pred = apply_error_correction(
            final_pred, 
            validation_df, 
            error_models, 
            target_col_list,
            strength=0.5  # 50%の強度で補正
        )
        
        # 補正後のスコア計算
        try:
            corrected_score = calc_score(validation_true, corrected_pred, target_col_list)
            print(f"\n補正後の予測スコア: {corrected_score:.6f}")
            print(f"改善率: {(corrected_score - basic_score) / basic_score * 100:.2f}%")
            
            # カテゴリ別スコアの計算
            for cat in target_col_list:
                cat_score = roc_auc_score(validation_true[cat], corrected_pred[cat])
                print(f"  {cat} 補正後スコア: {cat_score:.6f}")
                
        except Exception as e:
            print(f"補正後のスコア計算でエラーが発生しました: {e}")
            corrected_score = basic_score
            corrected_pred = final_pred.copy()
        
        # 閾値最適化を実行
        print("\n14. 閾値の最適化を実行...")
        optimal_thresholds = {}
        
        for cat in target_col_list:
            best_score = 0
            best_threshold = 0.5  # デフォルト閾値
            
            # データ型と範囲を確認
            print(f"カテゴリ {cat} の予測値チェック: min={corrected_pred[cat].min()}, max={corrected_pred[cat].max()}")
            
            try:
                # 様々な閾値を試す
                for threshold in np.linspace(0.01, 0.99, 50):
                    try:
                        # 閾値を適用した二値予測
                        binary_pred = (corrected_pred[cat] >= threshold).astype(float)
                        
                        # AUCスコアを計算
                        score = roc_auc_score(validation_true[cat], binary_pred)
                        
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                    except Exception as e:
                        continue
                
                optimal_thresholds[cat] = best_threshold
                print(f"カテゴリ {cat} の最適閾値: {best_threshold:.4f}, スコア: {best_score:.4f}")
            except Exception as e:
                print(f"カテゴリ {cat} の閾値最適化でエラー: {e}")
                optimal_thresholds[cat] = 0.5
        
        # 閾値を適用した予測値
        thresholded_pred = corrected_pred.copy()
        
        # 最適化後のスコアを計算
        try:
            threshold_score = calc_score(validation_true, thresholded_pred, target_col_list)
            print(f"閾値最適化後のスコア: {threshold_score:.6f}")
        except Exception as e:
            print(f"閾値最適化後のスコア計算でエラー: {e}")
            threshold_score = corrected_score
        
        return corrected_pred, error_models, optimal_thresholds
    
    # =================
    # 本番予測モードの処理
    # =================
    else:
        # NaN値が残っていないか確認
        nan_count = final_pred.isna().sum().sum()
        if nan_count > 0:
            print(f"警告: 最終予測に {nan_count} 個のNaN値が残っています。すべて0で置換します。")
            final_pred = final_pred.fillna(0)
        
        return final_pred

# =========================================
# メイン処理関数
# =========================================

def main():
    """
    メイン処理を実行する関数
    バリデーション → 本番予測 → 提出ファイル作成の流れを制御
    """
    print("===== ベイズ推定強化予測モデル v1.0 =====")
    print("- 時間的特徴量の拡充")
    print("- ベイズ推定による顧客購入確率の計算")
    print("- カテゴリごとの最適パラメータ調整")
    print("- カテゴリ購入シーケンスの特徴化")
    print("- フィードバックループによる予測精度向上")
    print("===========================================")
    
    # =================
    # データの読み込み
    # =================
    print("\nデータの読み込みを開始します...")
    train_session_df, test_session_df, train_target_df, train_log_df, jan_df = load_data()
    
    # =================
    # バリデーション実行
    # =================
    print("\nバリデーションを実行します...")
    try:
        corrected_pred, error_models, optimal_thresholds = bayes_enhanced_prediction_model(
            train_session_df, 
            test_session_df, 
            train_target_df, 
            train_log_df,
            jan_df,
            target_col_list,
            validate=True
        )
        
        # バリデーション予測を別ファイルとしても保存
        val_pred_filename = f'bayes_validation_pred_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        corrected_pred.to_csv(val_pred_filename, index=False)
        print(f"補正後バリデーション予測を {val_pred_filename} として保存しました。")
        
    except Exception as e:
        print(f"バリデーション中にエラーが発生しました: {e}")
        print("バリデーションをスキップして予測に進みます...")
        error_models = {}
        optimal_thresholds = {col: 0.5 for col in target_col_list}  # デフォルト閾値
    
    # =================
    # 本番予測の実行
    # =================
    print("\n最終予測モデルを実行します...")
    predictions = bayes_enhanced_prediction_model(
        train_session_df, 
        test_session_df, 
        train_target_df, 
        train_log_df,
        jan_df,
        target_col_list
    )
    
    # test_session_dfに必要な前処理が適用されていることを確認
    if 'enhanced_processed' not in test_session_df.columns:
        print("テストデータに前処理を適用します。")
        test_df_processed = enhanced_preprocess_data(test_session_df)
    else:
        test_df_processed = test_session_df
    
    # エラー修正モデルの適用
    if error_models:
        print("\nエラー修正モデルを適用します...")
        predictions = apply_error_correction(
            predictions, 
            test_df_processed,  # 前処理済みのテストデータを使用
            error_models, 
            target_col_list,
            strength=0.7  # 本番予測では少し強めに補正
        )
    
    # 閾値の適用（必要な場合）
    print("\n最適閾値を表示します...")
    for col in target_col_list:
        print(f"{col}の閾値: {optimal_thresholds.get(col, 0.5):.4f}")
        # 確率値をそのまま提出する場合はコメントアウト
        # predictions[col] = (predictions[col] >= optimal_thresholds.get(col, 0.5)).astype(float)
    
    # =================
    # 提出ファイルの作成
    # =================
    print("\n提出ファイルの作成を開始します...")
    submission = pd.DataFrame({
        'チョコレート': predictions['チョコレート'],
        'ビール': predictions['ビール'],
        'ヘアケア': predictions['ヘアケア'],
        '米（5㎏以下）': predictions['米（5㎏以下）']
    })
    
    # 現在の日時を取得してファイル名に含める
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_filename = f'submission_bayes_{now}.csv'
    
    # 提出ファイルの保存
    submission.to_csv(submission_filename, index=False)
    print(f"提出ファイルが {submission_filename} として保存されました。")
    
    # =================
    # 結果の確認
    # =================
    print("\n最終予測の先頭5件:")
    print(submission.head())
    
    # 各カテゴリの予測値の分布を確認
    print("\n各カテゴリの予測値の統計量:")
    print(submission.describe())

# =========================================
# 実行部分
# =========================================

if __name__ == "__main__":
    main()================
    # 日付・時間の前処理
    # =================
    df_copy['売上日'] = pd.to_datetime(df_copy['売上日'])
    df_copy['時刻'] = pd.to_datetime(df_copy['時刻'])
    
    # =================
    # 時間的特徴量の生成
    # =================
    
    # 基本的な時間特徴量
    df_copy['月'] = df_copy['売上日'].dt.month
    df_copy['日'] = df_copy['売上日'].dt.day
    
    # 曜日特徴 (0=月曜日, 6=日曜日)
    df_copy['曜日'] = df_copy['売上日'].dt.dayofweek
    
    # 週末フラグ（土日の購買パターンは平日と異なる可能性）
    df_copy['週末フラグ'] = df_copy['曜日'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 月内の週番号（給料日や特売日などの影響を捉える）
    df_copy['月内週番号'] = df_copy['日'].apply(lambda x: (x - 1) // 7 + 1)
    
    # 祝日フラグ（祝日の特別な購買パターンを捉える）
    df_copy['祝日フラグ'] = df_copy['売上日'].apply(lambda x: 1 if x in JP_HOLIDAYS else 0)
    
    # 時間帯区分（購買行動は時間帯によって大きく異なる）
    hour = df_copy['時刻'].dt.hour
    df_copy['時間帯'] = pd.cut(
        hour, 
        bins=[0, 6, 11, 14, 17, 21, 24], 
        labels=['深夜', '午前', '昼', '午後', '夕方', '夜']
    )
    
    # 時間帯をダミー変数化（機械学習で扱いやすくするため）
    time_dummies = pd.get_dummies(df_copy['時間帯'], prefix='時間帯')
    df_copy = pd.concat([df_copy, time_dummies], axis=1)
    
    # 月初・月末効果の考慮（給料日、支払日の影響）
    df_copy['月初フラグ'] = (df_copy['日'] <= 5).astype(int)    # 5日以内は月初
    df_copy['月末フラグ'] = (df_copy['日'] >= 25).astype(int)   # 25日以降は月末
    df_copy['月中フラグ'] = ((df_copy['日'] > 5) & (df_copy['日'] < 25)).astype(int)  # それ以外は月中
    
    # 特定の曜日パターン（例：給料日パターン）
    df_copy['第2金曜'] = ((df_copy['月内週番号'] == 2) & (df_copy['曜日'] == 4)).astype(int)
    df_copy['第4金曜'] = ((df_copy['月内週番号'] == 4) & (df_copy['曜日'] == 4)).astype(int)
    
    # =================
    # Recency特徴の生成
    # =================
    
    # 顧客の過去の購買履歴から最新購入日からの経過日数などを計算
    if train_log_df is not None and train_targets is not None:
        # セッションと顧客CDのマッピング辞書を作成
        session_customer_dict = df[['session_id', '顧客CD']].drop_duplicates().set_index('session_id')['顧客CD'].to_dict()
        
        # 各顧客の各カテゴリの購入日を時系列で整理
        customer_cat_purchase_dates = {}
        
        # ターゲットデータをマージ
        purchase_history = pd.merge(
            train_targets,
            df[['session_id', '売上日', '顧客CD']],
            on='session_id',
            how='left'
        )
        
        # 売上日をdatetime型に変換
        purchase_history['売上日'] = pd.to_datetime(purchase_history['売上日'])
        
        # 各顧客・各カテゴリの購入履歴を集計
        for cat in target_col_list:
            # 購入がある行のみフィルタ
            cat_purchases = purchase_history[purchase_history[cat] > 0][['顧客CD', '売上日']].copy()
            
            # 顧客ごとにグループ化して購入日のリストを作成
            for customer, group in cat_purchases.groupby('顧客CD'):
                # datetime型への変換を確実に実行
                dates = sorted(group['売上日'].dt.to_pydatetime().tolist())
                if customer not in customer_cat_purchase_dates:
                    customer_cat_purchase_dates[customer] = {}
                customer_cat_purchase_dates[customer][cat] = dates
        
        # 各顧客のカテゴリごとの最新購入日と平均購入間隔を計算
        customer_cat_recency = {}
        customer_cat_frequency = {}
        customer_cat_avg_interval = {}
        
        for customer, cat_dates in customer_cat_purchase_dates.items():
            if customer not in customer_cat_recency:
                customer_cat_recency[customer] = {}
                customer_cat_frequency[customer] = {}
                customer_cat_avg_interval[customer] = {}
                
            for cat, dates in cat_dates.items():
                if len(dates) > 0:
                    # 最新購入日
                    customer_cat_recency[customer][cat] = max(dates)
                    # 購入頻度
                    customer_cat_frequency[customer][cat] = len(dates)
                    
                    # 平均購入間隔の計算
                    if len(dates) > 1:
                        try:
                            intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                            customer_cat_avg_interval[customer][cat] = sum(intervals) / len(intervals)
                        except TypeError:
                            # エラー処理として安全な値を設定
                            print(f"タイプエラー: customer={customer}, cat={cat}, dates[0]の型={type(dates[0])}")
                            customer_cat_avg_interval[customer][cat] = 30
                    else:
                        customer_cat_avg_interval[customer][cat] = 30  # デフォルト値
        
        # Recency特徴を追加
        for cat in target_col_list:
            # 初期値として大きな値（未購入を意味する）を設定
            df_copy[f'{cat}_最終購入からの日数'] = 999
            df_copy[f'{cat}_購入頻度'] = 0
            df_copy[f'{cat}_平均購入間隔'] = 999
            
            # Recency計算（最終購入日からの経過日数）
            for idx, row in df_copy.iterrows():
                customer = row['顧客CD']
                current_date = row['売上日']
                
                # 最終購入からの日数
                if customer in customer_cat_recency and cat in customer_cat_recency[customer]:
                    last_purchase = customer_cat_recency[customer][cat]
                    days_since = (current_date - last_purchase).days
                    df_copy.at[idx, f'{cat}_最終購入からの日数'] = days_since
                
                # 購入頻度
                if customer in customer_cat_frequency and cat in customer_cat_frequency[customer]:
                    df_copy.at[idx, f'{cat}_購入頻度'] = customer_cat_frequency[customer][cat]
                
                # 平均購入間隔
                if customer in customer_cat_avg_interval and cat in customer_cat_avg_interval[customer]:
                    df_copy.at[idx, f'{cat}_平均購入間隔'] = customer_cat_avg_interval[customer][cat]
            
            # 購入サイクル近接度（平均間隔に対する最終購入からの日数の比率）
            df_copy[f'{cat}_サイクル近接度'] = df_copy.apply(
                lambda row: max(0, min(1, 
                                       row[f'{cat}_平均購入間隔'] / max(1, row[f'{cat}_最終購入からの日数'])
                                      )) if row[f'{cat}_平均購入間隔'] < 999 else 0, 
                axis=1
            )
    
    # =================
    # カテゴリ変数のエンコーディング
    # =================
    
    # 店舗名をダミー変数化
    store_dummies = pd.get_dummies(df_copy['店舗名'], prefix='店舗')
    df_copy = pd.concat([df_copy, store_dummies], axis=1)
    
    # 年代をダミー変数化
    age_dummies = pd.get_dummies(df_copy['年代'], prefix='年代')
    df_copy = pd.concat([df_copy, age_dummies], axis=1)
    
    # 性別をダミー変数化
    gender_dummies = pd.get_dummies(df_copy['性別'], prefix='性別')
    df_copy = pd.concat([df_copy, gender_dummies], axis=1)
    
    # 処理済みフラグを設定
    df_copy['enhanced_processed'] = 1
    
    return df_copy

# =========================================
# 評価指標計算
# =========================================

def calc_score(y_true, y_pred, target_col_list):
    """
    Macro AUC スコアを計算
    
    Args:
        y_true: 実測値のデータフレーム
        y_pred: 予測値のデータフレーム
        target_col_list: ターゲットカテゴリのリスト
    
    Returns:
        float: Macro AUC スコア
    """
    score_list = []
    for target_col in target_col_list:
        score = roc_auc_score(y_true[target_col], y_pred[target_col])
        score_list.append(score)
    return np.mean(score_list)

# =========================================
# シーケンス特徴と購入パターン分析
# =========================================

def add_sequence_features(train_session_df, train_target_df, target_col_list):
    """
    顧客の購入シーケンスに基づく特徴量を生成
    連続する購買行動のパターンから次の購買を予測する
    
    Args:
        train_session_df: セッション情報の訓練データ
        train_target_df: ターゲット情報の訓練データ
        target_col_list: ターゲットカテゴリのリスト
    
    Returns:
        tuple: (シーケンスパターン辞書, シーケンスから次のカテゴリへの確率辞書)
    """
    # セッションID、顧客CD、売上日を含むデータフレームを作成
    session_info = train_session_df[['session_id', '顧客CD', '売上日']].copy()
    session_info['売上日'] = pd.to_datetime(session_info['売上日'])
    
    # ターゲット情報をマージ
    purchase_history = pd.merge(
        train_target_df,
        session_info,
        on='session_id',
        how='left'
    )
    
    # 顧客ごとの購入履歴を時系列で整理
    customer_sequences = {}
    
    for customer, group in purchase_history.groupby('顧客CD'):
        # 日付でソート
        sorted_group = group.sort_values('売上日')
        
        # 各セッションでのカテゴリ購入情報を抽出
        session_purchases = []
        
        for _, row in sorted_group.iterrows():
            # そのセッションで購入したカテゴリのリストを作成
            categories = []
            for cat in target_col_list:
                if row[cat] > 0:
                    categories.append(cat)
            
            if categories:  # 何かカテゴリを購入していれば追加
                session_purchases.append({
                    'date': row['売上日'],
                    'categories': categories
                })
        
        # 顧客の購入シーケンスを保存
        if session_purchases:
            customer_sequences[customer] = session_purchases
    
    # シーケンスパターンの分析
    # パターンの形式: (cat1, cat2) -> 次のカテゴリの確率
    sequence_patterns = {}
    
    # 長さ2のシーケンスパターン（2回連続の購買から3回目を予測）
    for customer, purchases in customer_sequences.items():
        if len(purchases) < 3:  # 少なくとも3回の購入履歴が必要
            continue
        
        for i in range(len(purchases) - 2):
            # 2つ連続する購入のカテゴリの組み合わせを取得
            prev_cats1 = purchases[i]['categories']
            prev_cats2 = purchases[i+1]['categories']
            next_cats = purchases[i+2]['categories']
            
            # 各カテゴリの組み合わせについて集計
            for cat1 in prev_cats1:
                for cat2 in prev_cats2:
                    seq_key = (cat1, cat2)
                    
                    if seq_key not in sequence_patterns:
                        sequence_patterns[seq_key] = {'total': 0}
                        for target_cat in target_col_list:
                            sequence_patterns[seq_key][target_cat] = 0
                    
                    sequence_patterns[seq_key]['total'] += 1
                    
                    for next_cat in next_cats:
                        sequence_patterns[seq_key][next_cat] += 1
    
    # 各シーケンスパターンから次のカテゴリの条件付き確率を計算
    sequence_probabilities = {}
    
    for seq_key, counts in sequence_patterns.items():
        total = counts['total']
        if total > 0:  # 0除算を避ける
            sequence_probabilities[seq_key] = {}
            
            for cat in target_col_list:
                probability = counts[cat] / total
                sequence_probabilities[seq_key][cat] = probability
    
    # よく出現するパターンのみを残す（ノイズ低減）
    min_occurrence = 5  # 最低5回出現するパターンのみ残す
    filtered_patterns = {k: v for k, v in sequence_patterns.items() if v['total'] >= min_occurrence}
    filtered_probabilities = {k: v for k, v in sequence_probabilities.items() 
                             if k in filtered_patterns}
    
    return filtered_patterns, filtered_probabilities

def get_customer_latest_sequence(df, customer, customer_sequences, n=2):
    """
    顧客の最新n回の購入カテゴリを取得
    
    Args:
        df: 現在の予測対象データ
        customer: 顧客ID
        customer_sequences: 顧客ごとの購入シーケンス辞書
        n: 取得するシーケンスの長さ
    
    Returns:
        list: 最新n回の購入カテゴリのリスト（不足する場合はNone埋め）
    """
    if customer not in customer_sequences or len(customer_sequences[customer]) < n:
        return [None] * n
    
    # 顧客の購入履歴を取得
    purchases = customer_sequences[customer]
    
    # 最新n回の購入カテゴリを取得
    latest_categories = []
    for i in range(1, n+1):
        idx = len(purchases) - i
        if idx >= 0:
            # 複数カテゴリ購入の場合は最初のカテゴリを使用
            cats = purchases[idx]['categories']
            latest_categories.append(cats[0] if cats else None)
        else:
            latest_categories.append(None)
    
    # 新しい順から古い順に並べ替え
    return latest_categories[::-1]

def apply_sequence_features(df, customer_sequences, sequence_probabilities, target_col_list):
    """
    データフレームにシーケンス特徴を適用
    
    Args:
        df: 特徴を適用するデータフレーム
        customer_sequences: 顧客ごとの購入シーケンス辞書
        sequence_probabilities: シーケンスパターンから次のカテゴリへの確率辞書
        target_col_list: ターゲットカテゴリのリスト
    
    Returns:
        DataFrame: シーケンス特徴が追加されたデータフレーム
    """
    # 結果を格納するデータフレーム
    result_df = df.copy()
    
    # 各カテゴリのシーケンス確率を格納する列を初期化
    for cat in target_col_list:
        result_df[f'{cat}_シーケンス確率'] = 0.0
    
    # 各行に対してシーケンス特徴を適用
    for idx, row in result_df.iterrows():
        customer = row['顧客CD']
        
        # 顧客の最新の購入シーケンスを取得
        latest_sequence = get_customer_latest_sequence(df, customer, customer_sequences, n=2)
        
        # シーケンスが完全であれば確率を適用
        if None not in latest_sequence:
            seq_key = tuple(latest_sequence)
            if seq_key in sequence_probabilities:
                probs = sequence_probabilities[seq_key]
                
                for cat in target_col_list:
                    if cat in probs:
                        result_df.at[idx, f'{cat}_シーケンス確率'] = probs[cat]
    
    return result_df

# =========================================
# ベイズ推定による購入確率計算
# =========================================

def calculate_bayes_probabilities(training_df, predict_df, target_col_list, attribute_cols=None, confidence_params=None):
    """
    ベイズ推定を使用して顧客ごとの購入確率を計算
    事前分布（属性グループ）と事後分布（個人履歴）を組み合わせて予測
    
    Args:
        training_df: 訓練データ
        predict_df: 予測対象データ
        target_col_list: ターゲットカテゴリのリスト
        attribute_cols: 属性グループ化に使用するカラムのリスト
        confidence_params: 各カテゴリの確信度パラメータ
        
    Returns:
        DataFrame: ベイズ推定による予測確率を含むデータフレーム
    """
    if attribute_cols is None:
        attribute_cols = ['年代', '性別', '店舗名']
    
    if confidence_params is None:
        # デフォルトの確信度パラメータ
        confidence_params = {
            'チョコレート': 10.0,
            'ビール': 10.0,
            'ヘアケア': 10.0,
            '米（5㎏以下）': 10.0
        }
    
    # 結果を格納するデータフレーム
    result_df = pd.DataFrame(index=predict_df.index, columns=target_col_list)
    
    # 属性グループごとの購入確率を計算
    attr_mean_target = training_df.groupby(attribute_cols).agg(
        **{f"{col}_count": (col, 'count') for col in target_col_list},
        **{f"{col}_sum": (col, 'sum') for col in target_col_list}
    ).reset_index()
    
    # 顧客CDごとの購入回数と購入確率を計算
    customer_mean_target = training_df.groupby('顧客CD').agg(
        **{f"{col}_count_customer": (col, 'count') for col in target_col_list},
        **{f"{col}_sum_customer": (col, 'sum') for col in target_col_list}
    ).reset_index()
    
    # 予測データに属性グループの統計をマージ
    enriched_predict = predict_df.copy()
    enriched_predict = pd.merge(
        enriched_predict,
        attr_mean_target,
        on=attribute_cols,
        how='left'
    )
    
    # 予測データに顧客統計をマージ
    enriched_predict = pd.merge(
        enriched_predict,
        customer_mean_target,
        on='顧客CD',
        how='left'
    )
    
    # NaN値を0に置き換え
    for col in target_col_list:
        for suffix in ['_count', '_sum', '_count_customer', '_sum_customer']:
            col_name = f"{col}{suffix}"
            if col_name in enriched_predict.columns:
                enriched_predict[col_name] = enriched_predict[col_name].fillna(0)
    
    # ベイズ推定による確率計算（カテゴリごとに個別のパラメータを使用）
    for cat in target_col_list:
        # 確信度パラメータを取得
        c = confidence_params[cat]
        
        # 属性グループの事前確率を計算（0除算を避ける）
        enriched_predict[f'{cat}_prior'] = enriched_predict.apply(
            lambda row: row[f'{cat}_sum'] / max(1, row[f'{cat}_count']),
            axis=1
        )
        
        # ベイズ推定による確率計算
        # P(購入|顧客) = (顧客の購入数 + c*P(購入|属性)) / (顧客のセッション数 + c)
        enriched_predict[cat] = enriched_predict.apply(
            lambda row: (row[f'{cat}_sum_customer'] + c * row[f'{cat}_prior']) / 
                       (row[f'{cat}_count_customer'] + c),
            axis=1
        )
        
        # 顧客の履歴がない場合は属性グループの確率を使用
        mask = enriched_predict[f'{cat}_count_customer'] == 0
        enriched_predict.loc[mask, cat] = enriched_predict.loc[mask, f'{cat}_prior']
        
        # 属性グループの履歴もない場合はカテゴリの全体平均を使用
        overall_mean = training_df[cat].mean()
        mask = (enriched_predict[f'{cat}_count_customer'] == 0) & (enriched_predict[f'{cat}_count'] == 0)
        enriched_predict.loc[mask, cat] = overall_mean
        
        # 結果をコピー
        result_df[cat] = enriched_predict[cat].values
    
    return result_df

# =========================================
# Optunaによるパラメータ最適化
# =========================================

def optimize_confidence_params(training_df, validation_df, validation_true, target_col_list, n_trials=50):
    """
    Optunaを使用して各カテゴリのベイズ推定確信度パラメータを最適化
    
    Args:
        training_df: 訓練データ
        validation_df: 検証データ
        validation_true: 検証データの正解ラベル
        target_col_list: ターゲットカテゴリのリスト
        n_trials: Optunaの試行回数
        
    Returns:
        dict: 最適化された各カテゴリの確信度パラメータ
    """
    # 各カテゴリごとに最適化
    best_params = {}
    
    for cat in target_col_list:
        print(f"カテゴリ '{cat}' の確信度パラメータを最適化中...")
        
        # カテゴリごとの目的関数
        def objective(trial):
            c = trial.suggest_float(f"c_{cat}", 0.1, 100.0, log=True)
            
            # このカテゴリだけの確信度パラメータを設定
            confidence_params = {target: 10.0 for target in target_col_list}
            confidence_params[cat] = c
            
            # ベイズ推定で予測
            pred = calculate_bayes_probabilities(
                training_df,
                validation_df,
                [cat],  # このカテゴリだけを対象
                confidence_params=confidence_params
            )
            
            # スコア計算
            try:
                score = roc_auc_score(validation_true[cat], pred[cat])
                return score
            except Exception as e:
                print(f"警告: カテゴリ '{cat}' のスコア計算でエラー: {e}")
                return 0.5  # エラー時はデフォルト値
        
        # Optunaで最適化
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # 最適値を保存
        best_c = study.best_params[f"c_{cat}"]
        best_score = study.best_value
        best_params[cat] = best_c
        
        print(f"  ベストパラメータ c_{cat} = {best_c:.4f}, スコア = {best_score:.6f}")
    
    return best_params

# =========================================
# エラー修正モデル（フィードバックループ）
# =========================================

def create_error_correction_model(validation_df, predictions, true_values, target_col_list):
    """
    予測誤差を学習し、補正するためのモデルを作成
    バリデーションでの誤差パターンを分析して本番予測を改善する
    
    Args:
        validation_df: 検証データの特徴量
        predictions: 予測値
        true_values: 実際の値
        target_col_list: ターゲットカテゴリのリスト
    
    Returns:
        dict: エラー修正モデル（各特徴の補正係数を含む辞書）
    """
    error_models = {}
    
    for cat in target_col_list:
        # 予測エラーを計算
        error = true_values[cat] - predictions[cat]
        
        # 検証データに予測値とエラーを追加
        analysis_df = validation_df.copy()
        analysis_df[f'{cat}_予測値'] = predictions[cat].values
        analysis_df[f'{cat}_エラー'] = error.values
        
        # エラー修正モデルの作成（各特徴とエラーの関係を分析）
        correction_model = {}
        
        # 1. 年代別の予測エラー
        age_error = analysis_df.groupby('年代')[f'{cat}_エラー'].mean().to_dict()
        correction_model['年代'] = age_error
        
        # 2. 店舗別の予測エラー
        store_error = analysis_df.groupby('店舗名')[f'{cat}_エラー'].mean().to_dict()
        correction_model['店舗名'] = store_error
        
        # 3. 曜日別の予測エラー
        day_error = analysis_df.groupby('曜日')[f'{cat}_エラー'].mean().to_dict()
        correction_model['曜日'] = day_error
        
        # 4. 時間帯別の予測エラー
        time_error = analysis_df.groupby('時間帯')[f'{cat}_エラー'].mean().to_dict()
        correction_model['時間帯'] = time_error
        
        # 5. 予測値レンジ別のエラー
        # 予測値を10個のビンに分ける
        analysis_df['予測ビン'] = pd.qcut(analysis_df[f'{cat}_予測値'], 10, labels=False, duplicates='drop')
        pred_bin_error = analysis_df.groupby('予測ビン')[f'{cat}_エラー'].mean().to_dict()
        correction_model['予測ビン'] = pred_bin_error
        
        # 予測ビンの境界値を保存
        bin_edges = pd.qcut(analysis_df[f'{cat}_予測値'], 10, duplicates='drop', retbins=True)[1]
        correction_model['予測ビン境界'] = bin_edges.tolist()
        
        # エラー修正モデルを保存
        error_models[cat] = correction_model
    
    return error_models

def apply_error_correction(predictions, df, error_models, target_col_list, strength=0.5):
    """
    エラー修正モデルを使って予測を補正
    
    Args:
        predictions: 元の予測値
        df: 予測対象の特徴量
        error_models: エラー修正モデル
        target_col_list: ターゲットカテゴリのリスト
        strength: 補正の強さ（0〜1）
    
    Returns:
        DataFrame: 補正された予測値
    """
    corrected_predictions = predictions.copy()
    
    # 必要な特徴量が存在するか確認
    required_columns = ['年代', '店舗名', '曜日', '時間帯']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: エラー修正に必要なカラム {missing_columns} がデータフレームにありません。")
        print("前処理を適用します。")
        # 基本的な前処理を適用
        df = enhanced_preprocess_data(df)
    
    for cat in target_col_list:
        if cat in error_models:
            model = error_models[cat]
            
            # 予測の補正値を計算
            correction = np.zeros(len(df))
            
            # 1. 年代による補正
            if '年代' in df.columns and '年代' in model:
                for idx, row in df.iterrows():
                    age = row['年代']
                    if age in model['年代']:
                        correction[idx] += model['年代'][age]
            
            # 2. 店舗名による補正
            if '店舗名' in df.columns and '店舗名' in model:
                for idx, row in df.iterrows():
                    store = row['店舗名']
                    if store in model['店舗名']:
                        correction[idx] += model['店舗名'][store]
            
            # 3. 曜日による補正
            if '曜日' in df.columns and '曜日' in model:
                for idx, row in df.iterrows():
                    day = row['曜日']
                    if day in model['曜日']:
                        correction[idx] += model['曜日'][day]
            
            # 4. 時間帯による補正
            if '時間帯' in df.columns and '時間帯' in model:
                for idx, row in df.iterrows():
                    time = row['時間帯']
                    if time in model['時間帯']:
                        correction[idx] += model['時間帯'][time]
            
            # 5. 予測値レンジによる補正
            if '予測ビン' in model and '予測ビン境界' in model:
                bin_edges = model['予測ビン境界']
                
                for idx, pred in enumerate(predictions[cat]):
                    # 予測値がどのビンに属するかを決定
                    bin_idx = 0
                    for i, edge in enumerate(bin_edges[1:], 1):
                        if pred <= edge:
                            bin_idx = i - 1
                            break
                    
                    # ビンが見つかれば補正を適用
                    if bin_idx in model['予測ビン']:
                        correction[idx] += model['予測ビン'][bin_idx]
            
            # 補正値の平均を取り、強度係数を適用
            div_factor = sum([
                '年代' in df.columns and '年代' in model,
                '店舗名' in df.columns and '店舗名' in model,
                '曜日' in df.columns and '曜日' in model,
                '時間帯' in df.columns and '時間帯' in model,
                '予測ビン' in model and '予測ビン境界' in model
            ])
            
            if div_factor > 0:  # 0除算を避ける
                avg_correction = correction / div_factor
                
                # 補正を適用（強度で調整）
                corrected_predictions[cat] = predictions[cat] + avg_correction * strength
                
                # 0〜1の範囲に収める
                corrected_predictions[cat] = corrected_predictions[cat].clip(0, 1)
            else:
                print(f"警告: カテゴリ {cat} には適用可能な補正要素がありません。")
    
    return corrected_predictions

def analyze_error_patterns(validation_df, predictions, true_values, target_col_list):
    """
    予測エラーパターンを分析し、特徴量ごとのエラー分布を出力
    
    Args:
        validation_df: 検証データの特徴量
        predictions: 予測値
        true_values: 実際の値
        target_col_list: ターゲットカテゴリのリスト
    """
    for cat in target_col_list:
        print(f"\n==== {cat} のエラー分析 ====")
        
        # 予測エラーを計算
        error = true_values[cat] - predictions[cat]
        
        # 検証データに予測値とエラーを追加
        analysis_df = validation_df.copy()
        analysis_df[f'{cat}_予測値'] = predictions[cat].values
        analysis_df[f'{cat}_エラー'] = error.values
        analysis_df[f'{cat}_絶対誤差'] = np.abs(error.values)
        
        # 1. 全体的なエラー統計
        print(f"平均誤差: {error.mean():.4f}")
        print(f"平均絶対誤差: {np.abs(error).mean():.4f}")
        print(f"最大過剰予測: {error.min():.4f}")
        print(f"最大過小予測: {error.max():.4f}")
        
        # 2. 年代別のエラー
        age_error = analysis_df.groupby('年代')[f'{cat}_絶対誤差'].mean()
        print("\n年代別の平均絶対誤差:")
        for age, err in age_error.sort_values(ascending=False).items():
            print(f"  {age}: {err:.4f}")
        
        # 3. 店舗別のエラー
        store_error = analysis_df.groupby('店舗名')[f'{cat}_絶対誤差'].mean()
        print("\n店舗別の平均絶対誤差 (上位5件):")
        for store, err in store_error.sort_values(ascending=False).head(5).items():
            print(f"  {store}: {err:.4f}")
        
        # 4. 曜日別のエラー
        day_error = analysis_df.groupby('曜日')[f'{cat}_絶対誤差'].mean()
        print("\n曜日別の平均絶対誤差:")
        for day, err in day_error.items():
            day_name = ['月', '火', '水', '木', '金', '土', '日'][day]
            print(f"  {day_name}曜日: {err:.4f}")
        
        # 5. 時間帯別のエラー
        time_error = analysis_df.groupby('時間帯')[f'{cat}_絶対誤差'].mean()
        print("\n時間帯別の平均絶対誤差:")
        for time, err in time_error.sort_values(ascending=False).items():
            print(f"  {time}: {err:.4f}")

# =========================================
# メイン予測モデル
# =========================================

def bayes_enhanced_prediction_model(train_session_df, test_session_df, train_target_df, train_log_df, jan_df, target_col_list, validate=False):
    """
    ベイズ推定で強化された予測モデル
    
    Args:
        train_session_df: 訓練用セッションデータ
        test_session_df: テスト用セッションデータ
        train_target_df: 訓練用ターゲットデータ
        train_log_df: 商品購入ログデータ
        jan_df: JANコードマスタ
        target_col_list: ターゲットカテゴリのリスト
        validate: バリデーションモードかどうか
    
    Returns:
        DataFrame or tuple: 予測値のデータフレーム、またはバリデーションモードの場合は(予測値, エラー修正モデル, 閾値)のタプル
    """
    # =================
    # 1. 前処理
    # =================
    print("1. 拡張データ前処理を実行中...")
    train_df = enhanced_preprocess_data(
        train_session_df.copy(), 
        jan_df, 
        train_log_df, 
        train_target_df
    )
    test_df = enhanced_preprocess_data(
        test_session_df.copy(), 
        jan_df, 
        train_log_df
    )
    
    # 訓練データにターゲットを追加
    print(f"train_df行数: {len(train_df)}, train_target_df行数: {len(train_target_df)}")
    assert len(train_df) == len(train_target_df), "行数が一致しません"
    
    for col in target_col_list:
        train_df[col] = train_target_df[col].values
    
    # バリデーションモードの場合は訓練データを時間で分割
    if validate:
        print("バリデーションモードで実行します...")
        # 時間で分割 (7-9月をトレーニング, 10月を検証)
        training_df = train_df[train_df['月'].isin([7, 8, 9])].copy().reset_index(drop=True)
        validation_df = train_df[train_df['月'] == 10].copy().reset_index(drop=True)
        
        # 検証用データ
        validation_true = validation_df[target_col_list].copy()
        
        # 予測用のデータは検証用データ
        predict_df = validation_df.copy()
    else:
        # 本番予測モード
        training_df = train_df.copy()
        predict_df = test_df.copy()
    
    # =================
    # 2. シーケンス特徴の生成
    # =================
    print("2. 購入シーケンス特徴の生成...")
    sequence_patterns, sequence_probabilities = add_sequence_features(
        train_session_df,
        train_target_df,
        target_col_list
    )
    
    print(f"  シーケンスパターン数: {len(sequence_patterns)}")
    
    # 各顧客の購入履歴を時系列で整理（シーケンス特徴用）
    customer_sequences = {}
    purchase_history = pd.merge(
        train_target_df,
        train_session_df[['session_id', '顧客CD', '売上日']],
        on='session_id',
        how='left'
    )
    purchase_history['売上日'] = pd.to_datetime(purchase_history['売上日'])
    
    for customer, group in purchase_history.groupby('顧客CD'):
        # 日付でソート
        sorted_group = group.sort_values('売上日')
        
        # 各セッションでのカテゴリ購入情報を抽出
        session_purchases = []
        
        for _, row in sorted_group.iterrows():
            # そのセッションで購入したカテゴリのリストを作成
            categories = []
            for cat in target_col_list:
                if row[cat] > 0:
                    categories.append(cat)
            
            if categories:  # 何かカテゴリを購入していれば追加
                session_purchases.append({
                    'date': row['売上日'],
                    'categories': categories
                })
        
        # 顧客の購入シーケンスを保存
        if session_purchases:
            customer_sequences[customer] = session_purchases
    
    # =================
    # 3. ベイズ推定パラメータの最適化
    # =================
    if validate:
        print("3. ベイズ推定パラメータの最適化...")
        best_confidence_params = optimize_confidence_params(
            training_df, 
            validation_df, 
            validation_true, 
            target_col_list, 
            n_trials=30  # 試行回数
        )
    else:
        # 本番モードでは最適化済みのパラメータを使用
        best_confidence_params = {
            'チョコレート': 15.3,  # 例：最適化済みの値
            'ビール': 8.7,
            'ヘアケア': 12.1,
            '米（5㎏以下）': 6.4
        }
    
    # =================
    # 4. 時間重み付きベイズ推定
    # =================
    print("4. ベイズ推定による顧客ベース予測を計算...")
    
    # 最新の日付を取得
    max_date = training_df['売上日'].max()
    
    # 日付の差分を日数で計算（時間重み用）
    training_df['days_from_max'] = (max_date - training_df['売上日']).dt.days
    
    # カテゴリごとの時系列減衰率（改良版）
    decay_rates = {
        'チョコレート': 0.015,  # よりゆるやかな減衰
        'ビール': 0.025,       # 現状維持
        'ヘアケア': 0.04,      # より急な減衰（最近の購入により重点）
        '米（5㎏以下）': 0.008  # 非常にゆるやかな減衰（長期的傾向を重視）
    }
    
    # カテゴリごとの時間重みを計算
    for col in target_col_list:
        decay_rate = decay_rates[col]
        training_df[f'time_weight_{col}'] = np.exp(-decay_rate * training_df['days_from_max'])
    
    # =================
    # 5. ベイズ推定で強化された予測を計算
    # =================
    print("5. ベイズ推定強化予測の統合...")
    
    # 確信度パラメータを使ってベイズ推定による顧客ベース予測を計算
    attribute_cols = ['年代', '性別', '店舗名']
    bayes_customer_pred = calculate_bayes_probabilities(
        training_df, 
        predict_df, 
        target_col_list, 
        attribute_cols=attribute_cols,
        confidence_params=best_confidence_params
    )
    
    # =================
    # 6. シーケンス特徴の適用
    # =================
    print("6. シーケンスベースの予測確率を適用...")
    predict_df = apply_sequence_features(
        predict_df,
        customer_sequences,
        sequence_probabilities,
        target_col_list
    )
    
    # =================
    # 7. その他の属性ベース予測
    # =================
    print("7. その他の属性ベース予測の計算...")
    
    # 曜日ベースの予測
    day_mean_df = training_df.groupby(['曜日'])[target_col_list].mean().reset_index()
    
    # 時間帯ベースの予測
    time_mean_df = training_df.groupby(['時間帯'])[target_col_list].mean().reset_index()
    
    # 月初・月末ベースの予測
    month_period_mean = training_df.groupby(['月初フラグ', '月末フラグ'])[target_col_list].mean().reset_index()
    
    # 店舗ベースの予測
    store_mean_df = training_df.groupby('店舗名')[target_col_list].mean().reset_index()
    
    # =================
    # 8. カテゴリ間の相関計算
    # =================
    print("8. カテゴリ間の相関を計算...")
    category_correlations = {}
    
    for cat1 in target_col_list:
        for cat2 in target_col_list:
            if cat1 != cat2:
                # cat1を購入した場合にcat2も購入する確率
                cat1_purchases = training_df[training_df[cat1] > 0]
                if len(cat1_purchases) > 0:
                    prob = cat1_purchases[cat2].mean()
                    category_correlations[(cat1, cat2)] = prob
    
    # =================
    # 9. 予測の統合
    # =================
    print("9. 予測の統合を実行中...")
    
    # 属性ベースの予測
    attr_pred = pd.merge(
        predict_df[attribute_cols],
        training_df.groupby(attribute_cols)[target_col_list].mean().reset_index(),
        on=attribute_cols,
        how='left'
    )
    
    # 曜日ベースの予測
    day_pred = pd.merge(
        predict_df[['曜日']],
        day_mean_df,
        on=['曜日'],
        how='left'
    )
    
    # 時間帯ベースの予測
    time_pred = pd.merge(
        predict_df[['時間帯']],
        time_mean_df,
        on=['時間帯'],
        how='left'
    )
    
    # 月初・月末ベースの予測
    month_period_pred = pd.merge(
        predict_df[['月初フラグ', '月末フラグ']],
        month_period_mean,
        on=['月初フラグ', '月末フラグ'],
        how='left'
    )
    
    # 店舗ベースの予測
    store_pred = pd.merge(
        predict_df[['店舗名']],
        store_mean_df,
        on=['店舗名'],
        how='left'
    )
    
    # カテゴリの最適重みをスコアに基づいて設定
    # ベイズ推定ベースの予測の重みを増加
    category_weights = {
        'チョコレート': {
            'ベイズ顧客': 0.75, '属性': 0.10, '曜日': 0.04, '時間帯': 0.04, 
            '月初月末': 0.02, '店舗': 0.03, 'シーケンス': 0.02
        },
        'ビール': {
            'ベイズ顧客': 0.80, '属性': 0.07, '曜日': 0.04, '時間帯': 0.04, 
            '月初月末': 0.02, '店舗': 0.01, 'シーケンス': 0.02
        },
        'ヘアケア': {
            'ベイズ顧客': 0.60, '属性': 0.20, '曜日': 0.04, '時間帯': 0.04, 
            '月初月末': 0.02, '店舗': 0.08, 'シーケンス': 0.02
        },
        '米（5㎏以下）': {
            'ベイズ顧客': 0.65, '属性': 0.15, '曜日': 0.04, '時間帯': 0.02, 
            '月初月末': 0.04, '店舗': 0.08, 'シーケンス': 0.02
        }
    }
    
    # カテゴリ間の相関補正強度
    correlation_strengths = {
        'チョコレート': 0.18,
        'ビール': 0.05,
        'ヘアケア': 0.12,
        '米（5㎏以下）': 0.15
    }
    
    # カテゴリ間の特定の関連性を調整
    special_correlation_pairs = {
        ('ビール', 'チョコレート'): 0.15,
        ('チョコレート', 'ビール'): 0.10,
        ('チョコレート', 'ヘアケア'): 0.12,
        ('チョコレート', '米（5㎏以下）'): 0.08
    }
    
    # 予測の統合
    final_pred = pd.DataFrame(index=predict_df.index, columns=target_col_list)
    
    for col in target_col_list:
        # カテゴリ固有の重みを取得
        weights = category_weights[col]
        
        # 各予測値を取得
        bayes_vals = bayes_customer_pred[col].fillna(0).values  # ベイズ推定による顧客予測
        attribute_vals = attr_pred[col].fillna(0).values
        day_vals = day_pred[col].fillna(0).values
        time_vals = time_pred[col].fillna(0).values
        month_period_vals = month_period_pred[col].fillna(0).values
        store_vals = store_pred[col].fillna(0).values
        
        # シーケンス予測値（ある場合）
        if f'{col}_シーケンス確率' in predict_df.columns:
            sequence_vals = predict_df[f'{col}_シーケンス確率'].fillna(0).values
        else:
            sequence_vals = np.zeros(len(predict_df))
        
        # 各予測値の重みを適用した加重平均
        final_pred[col] = (
            bayes_vals * weights['ベイズ顧客'] +
            attribute_vals * weights['属性'] +
            day_vals * weights['曜日'] +
            time_vals * weights['時間帯'] +
            month_period_vals * weights['月初月末'] +
            store_vals * weights['店舗'] +
            sequence_vals * weights['シーケンス']
        )
    
    # =================
    # 10. カテゴリ間相関による補正
    # =================
    print("10. カテゴリ間の相関に基づく補正を適用...")
    for col in target_col_list:
        # 他カテゴリからの補正値を計算
        correction = np.zeros(len(final_pred))
        for other_col in target_col_list:
            if other_col != col and (other_col, col) in category_correlations:
                # 特定のカテゴリペアに対して特別な重みを適用
                if (other_col, col) in special_correlation_pairs:
                    special_weight = special_correlation_pairs[(other_col, col)]
                    correction += final_pred[other_col].values * category_correlations[(other_col, col)] * special_weight
                else:
                    # 通常のカテゴリ固有の補正強度を適用
                    correlation_weight = correlation_strengths[col]
                    correction += final_pred[other_col].values * category_correlations[(other_col, col)] * correlation_weight
        
        # 補正を適用
        final_pred[col] = final_pred[col] + correction
        # 0〜1の範囲に収める
        final_pred[col] = final_pred[col].clip(0, 1)
    
    # =
