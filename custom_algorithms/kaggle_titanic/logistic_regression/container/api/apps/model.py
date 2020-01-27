import json

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


class Model(object):
    def __init__(self, config_path, mode, allow_update=False):
        self.config_path = config_path
        self.clf = None
        self.allow_update = allow_update

        if mode not in ['train', 'pred']:
            raise KeyError('modeに"train", "pred"を指定してない．')
        self.mode = mode
        if mode == 'pred':
            self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """TODO: 他のmoduleでも利用しているので，Utilsクラスにまとめる"""
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        # 想定する例外
        if not isinstance(config, dict):
            errmsg = f'設定ファイルが{type(config)}型となっている．' \
                + 'dictでなければならない．'
            raise TypeError(errmsg)
        expected_keys = ['model_path', 'hyper_params']
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

    def _save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config,
                      f,
                      ensure_ascii=False,
                      encoding='utf8',
                      indent=4)

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
            self._validate_dict(hyper_parameters, expected_keys)
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
        except (ValueError, TypeError) as e:
            # hyper_parametersのvaluesがLogisticRegressionの想定したものでない場合
            print(f'{type(e)}: ', e)
            print('LogisticRegressoionのデフォルトパラメータ値を代替として設定．')
            self.clf = LogisticRegression()

    def train_with_cv(self,
                      dataset,
                      cv=5,
                      return_train_score=True,
                      return_estimator=True):
        # 使用オブジェクトの検証
        self._validate_dataset(dataset)
        if self.clf is None:
            raise TypeError('モデルが初期化またはロードされていない．')

        """
        オプション引数によるハイパーパラメータ更新OKな場合．　または，
        ハイパーパラメータ設定辞書に各種オプションが設定されてない場合は
        設定辞書の値を上書きする．
        """
        flg1 = 'cv' not in self.config['hyper_params']
        flg2 = 'return_train_score' not in self.config['hyper_params']
        flg3 = 'return_estimator' not in self.config['hyper_params']
        if self.allow_update or flg1:
            self.config['hyper_params']['cv'] = cv
        if self.allow_update or flg2:
            self.config['hyper_params']['return_train_score'] = \
                return_train_score
        if self.allow_update or flg3:
            self.config['hyper_params']['return_estimator'] = \
                return_estimator
        self._save_config(self.config_path)  # 設定ファイルも更新
        hyper_params = self.config['hyper_params']

        # 学習（公差検証）
        scores = cross_validate(
            self.clf,
            dataset['X'],
            dataset['y'],
            cv=hyper_params['cv'],
            return_train_score=hyper_params['return_train_score'],
            return_estimator=hyper_params['return_estimator'])
        """
        今回は，簡単のためCV中最も良いvalidationスコアが出たものを採用する．
        このあたりはタスクによって手法を適宜変えれば良い (e.g. 平均をとる)
        """
        best_idx = scores['test_score'].argmax()
        self.clf = scores['estimator'][best_idx]

    def predict(self, dataset):
        self._validate_dataset(dataset)
        self.clf = joblib.load(self.config['model_path'])
        dataset['y'] = self.clf.predict(dataset['X'])
        return dataset
