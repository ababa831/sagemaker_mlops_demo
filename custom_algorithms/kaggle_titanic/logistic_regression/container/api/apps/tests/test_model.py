from pathlib import Path
import shutil
from copy import deepcopy

import pytest
import numpy as np


class TestModel(object):
    """Modelクラス，各メソッドの単体テスト例
    （一部結合テストになっている部分もある）

    テストケースを日本語で記述するメリット:
        開発チームが日本人のみなら，そのままドキュメントになる
        また，PyTest実行時のテスト結果の解釈性が向上

    TODO: 必要に応じて追記する（特にinit_modelの例外処理あたり）
    """
    @pytest.fixture
    def mmock_t(self, do):
        from model import Model
        m = Model(do.valid_config_path, mode='train')
        return m

    @pytest.fixture
    def mmock_p(self, do):
        from model import Model
        m = Model(do.valid_config_path, mode='pred')
        return m

    @pytest.fixture
    def y_pred(self, do, mmock_p):
        dummy_predset = do.dummy_dataset
        dummy_predset['y'] = None

        return mmock_p.predict(dummy_predset)

    # init_modelメソッド
    def test_初期化したモデルがhyperparam指定通りならTrue(self, do, mmock_t):
        """
        モックからデフォルトのhyperparams値を一部変える
        テスト結果が偶然でないことを確認する意図
        """
        do.hyper_params['random_state'] = 520
        mmock_t.init_model(hyper_parameters=do.hyper_params)

        result = deepcopy(do.hyper_params)
        update_items = {
            'random_state': mmock_t.clf.random_state,
            'solver': mmock_t.clf.solver,
            'class_weight': mmock_t.clf.class_weight,
            'n_jobs': mmock_t.clf.n_jobs
        }
        result.update(update_items)
        assert result == do.hyper_params

    def test_hyperparam処理でexceptが出たときの代替値が想定通りならTrue(self, do, mmock_t):
        invalid_params = ['this is invalid Type']
        mmock_t.init_model(hyper_parameters=invalid_params)
        expected = {
            'random_state': 0,
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        result = {
            'random_state': mmock_t.clf.random_state,
            'solver': mmock_t.clf.solver,
            'class_weight': mmock_t.clf.class_weight,
            'n_jobs': mmock_t.clf.n_jobs
        }
        assert expected == result

    # train_with_cvメソッド
    def test_オプション引数が公差検証に反映されていればTrue(self, do, mmock_t):
        mmock_t.init_model()

        # 辞書に設定されていると辞書の設定値を優先するが，このテストケースでは邪魔なので削除
        mmock_t.config['hyper_params'].pop('cv')

        expected_cv = 2
        mmock_t.train_with_cv(do.dummy_dataset, cv=expected_cv)
        result_cv = len(mmock_t.scores['estimator'])
        assert result_cv == expected_cv

    def test_設定辞書に公差検証paramを与えてそれが反映されていればTrue(self, do, mmock_t):
        mmock_t.init_model()
        mmock_t.train_with_cv(do.dummy_dataset)
        expected = do.hyper_params['cv']
        result_cv = len(mmock_t.scores['estimator'])
        assert result_cv == expected

    # save_modelメソッド
    @pytest.mark.parametrize('expected', ['./hoge', './a/b', '/a'])
    def test_指定したdst_dirにモデルを保存できればTrue(self, do, mmock_t, expected):
        mmock_t.clf = 'dummy_model'
        mmock_t.save_model(dst_dir=expected)

        expected_path = \
            Path(expected).joinpath(mmock_t.config['model_path'])
        is_expected_path_exists = expected_path.exists

        shutil.rmtree(expected)  # テスト用に作成したオブジェクトを削除

        assert is_expected_path_exists, '指定したdst_dirに保存できていない'

    # predictメソッド
    def test_推論値のshapeが想定通りであればTrue(self, do, y_pred):
        expected = (y_pred['X'].shape[0], )
        assert y_pred['y'].shape == expected

    def test_推論値が0または1であればTrue(self, do, y_pred):
        result_unique = np.unique(y_pred['y'])
        expected = np.array([0, 1])
        assert (result_unique == expected).all()
