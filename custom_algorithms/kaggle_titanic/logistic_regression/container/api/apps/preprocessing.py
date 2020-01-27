import warnings
import json

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

warnings.filterwarnings('ignore')


class PreProcessor(object):
    def __init__(self, config_path, mode, label='Survived'):
        self.label = label
        self.config_path = config_path  # 学習時の設定ファイル
        self.dataset = {'X': None, 'y': None}

        assert mode in ['train', 'pred']
        self.mode = mode
        if mode == 'pred':
            self.config = self._load_config(config_path)
            self.transformers = self._load_transformers(self.config)
        else:
            self.transformers = {
                'fillna_vals': {},
                'onehot_encoders': {},
                'count_corresp_tables': {},
                'minmax_scaler': None
            }

    def _load_transformers(self, config):
        """保存したログからtransformers 辞書を取得
        """
        transformers = joblib.load(config['transformer_paths'])
        expected_keys = [
            'fillna_vals', 'onehot_encoders', 'count_corresp_tables',
            'minmax_scaler'
        ]
        self._validate_dict(transformers, expected_keys)

        return transformers

    def _load_config(self, config_path):
        """
        TODO: 他のmoduleでも利用しているので，Utilsクラスにまとめる
        """
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        # 想定する例外
        if not isinstance(config, dict):
            errmsg = f'設定ファイルが{type(config)}型となっている．' \
                + 'dictでなければならない．'
            raise TypeError(errmsg)
        expected_keys = ['transformer_paths']
        self._validate_dict(config, expected_keys)

        return config

    def _validate_dict(self, dict_obj, expected_keys):
        """
        TODO: 他のmoduleでも利用しているので，Utilsクラスにまとめる
        """
        for exp_key in expected_keys:
            if exp_key not in dict_obj.keys():
                errmsg = f'{exp_key}が設定ファイルに含まれてない．'
                raise KeyError(errmsg)

    def get_dataset(self, input_df):
        # 入力データのチェック
        if isinstance(input_df, pd.core.frame.DataFrame):
            raise TypeError('入力データはDataFrame型のみ有効')

        # 学習用データでなければ（推論データであれば），学習時のmodelオブジェクトをロード
        self.is_train_set = self.label in input_df.columns
        if not self.is_train_set:
            self._load_transformers()

        # 特徴量エンジニアリング
        input_df = self.do_feature_engineering(input_df)

        # データセット作成
        cols_explanatory = [
            'Cabin_count', 'Fare', 'Pclass', 'Sex_female', 'Sex_male',
            'Embarked_C', 'Embarked_NA', 'Embarked_Q', 'Embarked_S'
        ]
        self.dataset['X'] = input_df[cols_explanatory].values
        self.dataset['y'] = input_df[self.label] if self.is_train_set else None

        return self.dataset

    def do_feature_engineering(self, input_df):
        # 欠損値処理（ざっくり）
        # 量的変数
        cols_quantitative = ['Age', 'SibSp', 'Parch', 'Fare']
        for col in cols_quantitative:
            fillna_val = None
            if self.is_train_set:
                fillna_val = input_df[col].median()  # To median
                # 後で推論データの前処理に適用するので，dataset辞書に記録
                self.transformers['fillna_vals'][col] = \
                    fillna_val
            else:
                fillna_val = self.transformers['fillna_vals'][col]
            input_df[col] = input_df[col].fillna(fillna_val)
        # カテゴリカル変数
        cols_categorical = ['Embarked', 'Cabin']
        for col in cols_categorical:
            input_df[col] = input_df[col].fillna('NA')

        # One-hot encoding
        cols_categ = ['Sex', 'Embarked']
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        for col in cols_categ:
            if not self.is_train_set:
                ohe = self.transformers['onehot_encoder'][col]

            # Encoding(ユニーク数分DataFrameにカラムを用意）
            uq = [f'{col}_{uq_val}' for uq_val in input_df[col].unique()]
            onehot_encoded = ohe.fit_transform(input_df[[col]])
            ohe_df = pd.DataFrame(onehot_encoded, columns=uq)

            # One-hot Encoding変換後のデータshapeが正しいか確認
            n_sample, n_target_col = input_df.shape[0], len(uq)
            assert ohe_df.shape == (n_sample, n_target_col)

            # 更新
            input_df = pd.concat([input_df, ohe_df], axis=1)

            if self.is_train_set:
                self.transformers['onehot_encoders'][col] = ohe

        # カウント変数の作成
        cols_to_count = ['Ticket', 'Cabin']
        for col in cols_to_count:
            tmp_counts = None
            if self.is_train_set:
                tmp_counts = input_df[col].value_counts().reset_index()
                tmp_counts.columns = [col, f'{col}_count']
                self.transformers['count_corresp_tables'][col] = tmp_counts
            else:
                tmp_counts = self.transformers['count_corresp_tables'][col]

            input_df = pd.merge(input_df, tmp_counts, on=col, how='left')

        # 裾が重い変数を対数変換
        errmsg = 'Cabin_countが入力データに存在していない'
        assert 'Cabin_count' in input_df.columns, errmsg
        cols_to_log1p = ['Fare', 'Cabin_count']
        for col in cols_to_log1p:
            input_df[col] = input_df[col].apply(np.log1p)

        # スケーリング
        counts = [f'{col}_count' for col in cols_to_count]
        cols_quantitative += counts
        scaler = MinMaxScaler()
        if self.is_train_set:
            input_df[cols_quantitative] = scaler.fit(
                input_df[cols_quantitative])
            self.transformers['minmax_scaler'] = scaler
        else:
            scaler = self.transformers['minmax_scaler']
        input_df[cols_quantitative] = \
            scaler.transform(input_df[cols_quantitative])

        return input_df
