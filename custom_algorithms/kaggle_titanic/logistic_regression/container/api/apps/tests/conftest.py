from pathlib import Path
import sys
import os
import json
import shutil
import joblib

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

sd = Path(__file__).parent.resolve()
sys.path.append(sd)
sys.path.append(sd.parent)


class DummyObjects(object):
    def __init__(self):
        self.data_dir = sd.joinpath('data')

        self.pred_model_dir = Path('/opt/ml/model')
        os.makedirs(self.pred_model_dir, exist_ok=True)

    def create_dummy_transformers(self):
        dummy_transformers_name = 'dummy_transformers'
        self.dummy_transformes_path = \
            self.pred_model_dir.joinpath(dummy_transformers_name)

        src_transfromers = self.data_dir.joinpath('dummy_trans.pkl.cmp')
        shutil.copyfile(src_transfromers, self.dummy_transformes_path)

    def create_valid_dummy_df(self):
        data_path = self.data_dir.joinpath('data_valid.csv')
        self.dummy_valid_df = pd.read_csv(data_path)

    def create_invalid_dummy_df(self):
        cols = self.dummy_valid_df.columns
        self.col_lacked_df = self.dummy_valid_df.drop(cols[-1], axis=1)

    def create_hyper_params(self):
        self.hyper_params = {
            'random_state': 0,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'n_jobs': -1,
            'cv': 5,
            'return_train_score': True
        }

    def create_dummy_dataset(self):
        self.dummy_dataset = {
            'X':
            np.array([[0., 0., 3., 1., 0., 1., 0.],
                      [0., 1., 1., 0., 1., 0., 1.],
                      [0., 0., 2., 1., 0., 1., 0.],
                      [0., 2., 3., 1., 0., 1., 0.],
                      [0., 0., 3., 1., 0., 1., 0.],
                      [0., 1., 1., 0., 1., 0., 1.],
                      [0., 0., 2., 1., 0., 1., 0.],
                      [0., 2., 3., 1., 0., 1., 0.],
                      [1., 0., 3., 1., 0., 0., 0.]]),
            'y':
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])
        }

    def create_dummy_model(self):
        dummy_model_name = 'dummy_model.pkl.cmp'
        trained_path = self.data_dir.joinpath(dummy_model_name)

        self.create_hyper_params()
        self.create_dummy_dataset()

        if not trained_path.exists:
            self.clf = LogisticRegression(
                random_state=self.hyper_params['random_state'],
                solver=self.hyper_params['solver'],
                class_weight=self.hyper_params['class_weight'],
                n_jobs=self.hyper_params['n_jobs'])
            self.clf.fit(self.dummy_dataset['X'], self.dummy_dataset['y'])

            joblib.dump(self.clf, trained_path, compress=True)

        # 推論用ディレクトリにコピーしておく（推論時のテストで利用）
        self.dummy_model_path = \
            self.pred_model_dir.joinpath(dummy_model_name)
        shutil.copyfile(trained_path, self.dummy_model_path)

    def create_dummy_config(self):
        """
        単体テスト用のデータ生成->
        テスト対象のメソッドを利用してデータを作成しないように注意
        """
        self.valid_config_path = \
            self.data_dir.joinpath('config_valid.json')

        # TODO:ダミーコンフィグデータ作成
        self.dummy_config = {
            'config_path': str(self.valid_config_path),
            'hyper_params': self.hyper_params,
            'transformer_paths': str(self.dummy_transformes_path),
            'model_path': str(self.dummy_model_path)
        }

        with open(self.valid_config_path, 'w', encoding='utf8') as f:
            json.dump(self.dummy_config, f, ensure_ascii=False, indent=4)


@pytest.fixture
def do():
    do = DummyObjects()
    do.create_dummy_transformers()
    do.create_valid_dummy_df()
    do.create_dummy_model()
    do.create_dummy_config()
    return do
