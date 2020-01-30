from pathlib import Path
import sys
import shutil

import pytest

sd = Path(__file__).parents[1].resolve()
sys.path.append(sd)


class TestPreProcessor(object):
    """PreProcessorクラス，各メソッドの単体テスト例
    （一部結合テストになっている部分もある）

    テストケースを日本語で記述するメリット:
        開発チームが日本人のみなら，そのままドキュメントになる
        また，PyTest実行時のテスト結果の解釈性が向上

    TODO: 必要に応じて追記する（特に特徴量エンジニアリングあたり）
    """
    @pytest.fixture
    def ppmock(self, do):
        from preprocessing import PreProcessor
        pp = PreProcessor(do.valid_config_path, mode='train')
        pp.config = do.dummy_config
        return pp

    @pytest.mark.parametrize('mode', ['train', 'pred'])
    def test_mode引数を正しく指定できればinstance生成成功(self, do, mode):
        from preprocessing import PreProcessor

        pp = PreProcessor(do.valid_config_path, mode=mode)
        assert pp

    def test_不正なmode引数ならinstance生成失敗でValueErrorを返す(self, do):
        from preprocessing import PreProcessor
        try:
            pp_invalid_mode = PreProcessor(do.valid_config_path,
                                           mode='invalid')
        except ValueError:
            pp_invalid_mode = None
        assert pp_invalid_mode is None

    @pytest.mark.parametrize('expected', ['./hoge', './a/b', '/a'])
    def test_transformersが指定したdst_dirに保存できればTrue(self, do, ppmock, expected):
        ppmock.save_transformers(dst_dir=expected)

        expected_path = \
            Path(expected).joinpath(ppmock.config['transformer_paths'])
        is_expected_path_exists = expected_path.exists

        shutil.rmtree(expected)  # テスト用に作成したオブジェクトを削除

        assert is_expected_path_exists, '指定したdst_dirに保存できていない'

    def test_学習時get_datasetで辞書型にyラベルが正しく存在すればTrue(self, do, ppmock):
        dummy_df = do.dummy_valid_df
        dataset = ppmock.get_dataset(dummy_df)
        errmsg = 'datasetにyが正しく与えられてない'
        assert (dataset['y'] == dummy_df['Survived'].values).all(), errmsg

    def test_推論時にget_datasetで辞書型にXが想定サンプル分ndarrayがあればTrue(self, do):
        from preprocessing import PreProcessor
        pp = PreProcessor(do.valid_config_path, mode='pred')
        dummy_df = do.dummy_valid_df
        dataset = pp.get_dataset(dummy_df)

        print('\n', dataset)

        errmsg = '特徴量Xのサンプル数が入力データのサンプル数と一致しない'
        assert dataset['X'].shape[0] == dummy_df.values.shape[0], errmsg
