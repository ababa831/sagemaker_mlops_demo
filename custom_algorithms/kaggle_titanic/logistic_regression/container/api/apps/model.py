import traceback
from pathlib import Path
import os
import joblib

import numpy as np
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
        expected_keys = []
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
                raise TypeError(f'{hyper_parameters}がdictでない．')
            expected_keys = [
                'random_state', 'solver', 'class_weight', 'n_jobs'
            ]
            Utils.validate_dict(hyper_parameters, expected_keys)
            isinstance(hyper_parameters['random_state'], int)
            isinstance(hyper_parameters['solver'], str)
            isinstance(hyper_parameters['class_weight'], str)
            isinstance(hyper_parameters['n_jobs'], int)
        except (TypeError, KeyError):
            """
            configに'hyper_params'キーとそのvaluesにexpected_keys
            が存在しない場合
            """
            traceback.print_exc()
            hyper_parameters = {
                'random_state': 0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_jobs': -1
            }

        # モデルの初期化
        self.clf = LogisticRegression(
            random_state=hyper_parameters['random_state'],
            solver=hyper_parameters['solver'],
            class_weight=hyper_parameters['class_weight'],
            n_jobs=hyper_parameters['n_jobs'])

    def train_with_cv(self, dataset, cv=4, return_train_score=True):
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
        self.scores = cross_validate(self.clf,
                                     dataset['X'],
                                     dataset['y'],
                                     cv=cv,
                                     return_train_score=return_train_score,
                                     return_estimator=True)
        """
        今回は，簡単のためCV中最も良いvalidationスコアが出たものを採用する．
        このあたりはタスクによって手法を適宜変えれば良い (e.g. 平均をとる)
        """
        best_idx = self.scores['test_score'].argmax()
        self.clf = self.scores['estimator'][best_idx]

    def save_model(self,
                   dst_dir='./.models',
                   child_dir=None,
                   model_name='logistic_regression.pkl.cmp'):
        self.dst_dir = dst_dir
        if not self.clf:
            print('モデルが学習またはロードされていないので保存しない')
            return

        dst_dir = Path(dst_dir).resolve()
        if child_dir is None:
            # '{acitve branchのHEAD commit ID}.pkl.cmp'のように表示
            repo_abspath = Path(__file__).resolve().parents[6]
            repo = Repo(repo_abspath)
            child_dir = repo.active_branch.commit.hexsha
        dst_path = dst_dir.joinpath(child_dir, model_name)

        if not dst_path.parent.exists():
            os.makedirs(dst_path.parent)

        joblib.dump(self.clf, dst_path, compress=True)
        print(dst_path, 'にモデルを保存')

        # 子ディレクトリ以下のパスを記録（推論時に使用）
        self.config['model_path'] = \
            Path(child_dir).joinpath(model_name)
        self.cm.save_config(self.config, self.config_path)
        print(f'モデル保存先を設定ファイル{self.config_path}を更新')

    def predict(self, dataset):
        """入力データセット内'X'に対する推論結果yをデータセットに付与して返す
        
        Parameters
        ----------
        dataset : dict
            前処理・特徴量エンジニアリング済みデータセット
            {'X': shape(サンプル数, 変数の数), 'y': shape(サンプル数, )}
        
        Returns
        -------
        dict
            推論結果'y'が更新されたデータセット
        """        
        prefix = '/opt/ml/model'
        model_path_for_pred = Path(prefix).joinpath(self.config['model_path'])

        self._validate_dataset(dataset)

        self.clf = joblib.load(model_path_for_pred)
        dataset['y'] = self.clf.predict(dataset['X'])

        return dataset
