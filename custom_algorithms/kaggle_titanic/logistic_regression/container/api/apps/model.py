import traceback
import sys
from pathlib import Path
import os

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from git import Repo

from utils import Utils
from config_manager import ConfigManager


class Model(object):
    def __init__(self, config_path, mode):
        self.config_path = config_path
        self.clf = None
        self.cm = ConfigManager()

        if mode not in ['train', 'pred']:
            raise ValueError('modeに"train", "pred"を指定してない．')
        self.mode = mode
        if mode == 'pred':
            expected_keys = ['model_path', 'hyper_params']
            self.config = \
                self.cm.load_config(config_path, expected_keys)

    def _validate_dataset(self, dataset):
        if not isinstance(dataset, dict):
            raise TypeError('入力データセットがdictでない．')
        if 'X' not in dataset:
            raise KeyError('データセットに key: "X" が含まれていない')
        if self.mode == 'train' and 'y' not in dataset:
            raise KeyError('データセットに key: "y" が含まれていない')
        if not isinstance(dataset['X'], np.ndarray):
            raise TypeError('Xのvalueがarrayでない')
        if self.mode == 'train' and not isinstance(dataset['y'], np.ndarray):
            raise TypeError('yのvalueがarrayでない')

    def init_model(self, hyper_parameters=None):
        # ハイパーパラメータが引数に渡されなかった場合は，configから読み込む
        if hyper_parameters is None:
            hyper_parameters = self.config['hyper_params']

        # ハイパーパラメータ辞書の検証
        try:
            if not isinstance(hyper_parameters, dict):
                raise KeyError(f'{hyper_parameters}がdictでない．')
            expected_keys = [
                'random_state', 'solver', 'class_weight', 'n_jobs'
            ]
            Utils.validate_dict(hyper_parameters, expected_keys)
        except KeyError:
            """
            configに'hyper_params'キーとそのvaluesにexpected_keys
            が存在しない場合
            """
            hyper_parameters = {
                'random_state': 0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_jobs': -1
            }

        # モデルの初期化
        try:
            self.clf = LogisticRegression(
                random_state=hyper_parameters['random_state'],
                solver=hyper_parameters['solver'],
                class_weight=hyper_parameters['class_weight'],
                n_jobs=hyper_parameters['n_jobs'])
        except (ValueError, TypeError):
            errmsg = 'hyper_parametersのvaluesが' \
                + 'LogisticRegressionの想定した値・型でない場合'
            traceback.print_exc()
            sys.exit(errmsg)

    def train_with_cv(self, dataset, cv=5, return_train_score=True):
        # 使用オブジェクトの検証
        self._validate_dataset(dataset)
        if self.clf is None:
            raise TypeError('モデルが初期化またはロードされていない．')
        """
        configに特定のcv, return_train_score
        が指定されていたらオプション値を更新
        """
        if 'cv' in self.config['hyper_params']:
            cv = self.config['hyper_params']['cv']
        if 'return_train_score' in self.config['hyper_params']:
            return_train_score = \
                self.config['hyper_params']['return_train_score']

        # 学習（公差検証）
        scores = cross_validate(
            self.clf,
            dataset['X'],
            dataset['y'],
            cv=cv,
            return_train_score=return_train_score,
            return_estimator=True)
        """
        今回は，簡単のためCV中最も良いvalidationスコアが出たものを採用する．
        このあたりはタスクによって手法を適宜変えれば良い (e.g. 平均をとる)
        """
        best_idx = scores['test_score'].argmax()
        self.clf = scores['estimator'][best_idx]

    def save_model(self,
                   dst_dir='./.models',
                   child_dir=None,
                   model_name='logistic_regression.pkl.cmp'):
        if not self.clf:
            print('モデルが学習またはロードされていないので保存しない')
            return

        dst_dir = Path(dst_dir).resolve()
        if child_dir is None:
            # '{acitve branchのHEAD commit ID}.pkl.cmp'のように表示
            repo_abspath = Path(__file__).resolve().parents[6]
            repo = Repo(repo_abspath)
            child_dir = repo.active_branch.commit.hexsha
        dst_path = dst_dir.joinpath([child_dir, model_name])

        if not dst_path.parent.exists():
            os.makedirs(dst_path.parent)

        joblib.dump(self.clf, dst_path, compress=True)
        print(dst_path, 'にモデルを保存')

        self.config['model_path'] = dst_path
        self.cm.save_config(self.config, self.config_path)
        print(f'モデル保存先を設定ファイル{self.config_path}に上書き')

    def predict(self, dataset):
        self._validate_dataset(dataset)
        self.clf = joblib.load(self.config['model_path'])
        dataset['y'] = self.clf.predict(dataset['X'])
        return dataset
