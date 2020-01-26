import warnings

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

warnings.filterwarnings('ignore')


class PreProcessor(object):
    def __init__(self, label='Survived', config_path=None):
        self.label = label
        self.config_path = config_path  # 学習時の設定ファイル
        self.dataset = {'X': None, 'y': None}
        self.model_objects = {
            'fillna_vals': {},
            'onehot_encoders': {},
            'count_corresp_tables': {},
            'minmax_scaler': None
        }

        if config_path:
            self.config = self._load_config()

    def _load_model_objects(self):
        """TODO: 以下を実装
        保存したログからmodel_objects 辞書を取得
        def _load_config　を用意してコンフィグを読み込む
        e.g. model_obj_path = config['model_obj_path']
        model_objects = joblib.load(model_obj_path)
        """
        raise NotImplementedError

    def _load_config(self):
        pass

    def get_dataset(self, input_df):
        """
        """
        # 入力データのチェック
        if isinstance(input_df, pd.core.frame.DataFrame):
            raise TypeError('入力データはDataFrame型のみ有効')

        # 学習用データでなければ（推論データであれば），学習時のmodelオブジェクトをロード
        self.is_train_set = self.label in input_df.columns
        if not self.is_train_set:
            self._load_model_objects()

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
                self.model_objects['fillna_vals'][col] = \
                    fillna_val
            else:
                fillna_val = self.model_objects['fillna_vals'][col]
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
                ohe = self.model_objects['onehot_encoder'][col]

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
                self.model_objects['onehot_encoders'][col] = ohe

        # カウント変数の作成
        cols_to_count = ['Ticket', 'Cabin']
        for col in cols_to_count:
            tmp_counts = None
            if self.is_train_set:
                tmp_counts = input_df[col].value_counts().reset_index()
                tmp_counts.columns = [col, f'{col}_count']
                self.model_objects['count_corresp_tables'][col] = tmp_counts
            else:
                tmp_counts = self.model_objects['count_corresp_tables'][col]

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
            self.model_objects['minmax_scaler'] = scaler
        else:
            scaler = self.model_objects['minmax_scaler']
        input_df[cols_quantitative] = \
            scaler.transform(input_df[cols_quantitative])

        return input_df
