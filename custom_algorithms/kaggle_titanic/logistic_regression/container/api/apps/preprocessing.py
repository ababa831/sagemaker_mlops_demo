import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder


class PreProcessor(object):
    def __init__(self, label='Survived', config_path=None):
        self.label = label
        self.config_path = config_path  # 学習時の設定ファイル
        self.dataset = {
            'X': None,
            'y': None,
            'model_objects': {'fillna_vals': {}}
        }

    def _load_model_objects(self):
        """TODO: 以下を実装
        保存したログからmodel_objects 辞書を取得
        def _load_config　を用意してコンフィグを読み込む
        e.g. model_obj_path = config['model_obj_path']
        model_objects = joblib.load(model_obj_path)
        """
        raise NotImplementedError

    def get_dataset(self, input_df):
        if isinstance(input_df, pd.core.frame.DataFrame):
            raise TypeError('入力データはDataFrame型のみ有効')

        # 学習用データでなければ（推論データであれば），学習時のmodelオブジェクトをロード
        self.is_train_set = self.label in input_df.columns
        if not self.is_train_set:
            self._load_model_objects()
        
        self.do_feature_engineering(input_df)
            
        raise NotImplementedError

    def do_feature_engineering(self, input_df):
        # 欠損値処理（ざっくり）
        # 量的変数
        cols_quantitative = ['Age', 'SibSp', 'Parch', 'Fare']
        for col in cols_quantitative:
            fillna_val = None
            if self.is_train_set:
                fillna_val = input_df[col].median()  # To median
                # 後で推論データの前処理に適用するので，dataset辞書に記録
                self.dataset['model_objects']['fillna_vals'][col] = \
                    fillna_val
            else:
                fillna_val = self.dataset['model_objects']['fillna_vals'][col]
            input_df[col] = input_df[col].fillna(fillna_val)
        # カテゴリカル変数
        cols_categorical = ['Embarked', 'Cabin']
        for col in cols_categorical:
            input_df[col] = input_df[col].fillna('NA')
        
        # Encoding
        cols_categ = ['Sex', 'Embarked']
        cols_to_count = ['Ticket', 'Cabin']
        # One-hot encoding
        ohe = None
        if self.is_train_set:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            # TODO:ここにoheをfitさせてdfに突っ込む処理を書く
        else:
            ohe = self.dataset['model_objects']['onehot_encoder']

        raise NotImplementedError

