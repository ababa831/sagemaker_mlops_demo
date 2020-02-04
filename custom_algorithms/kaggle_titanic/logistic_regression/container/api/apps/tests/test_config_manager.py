from pathlib import Path
import json
import os
import time
import sys

import pytest

sd = Path(__file__).parent.resolve()
sys.path.append(sd.parent)


class TestConfigManager(object):
    @pytest.fixture
    def cmmock(self):
        from config_manager import ConfigManager
        return ConfigManager()

    @pytest.fixture
    def conf(self):
        dummy_config = {'a': 1, 'b': 2}
        config_path = sd.joinpath('data/dummy_conf.json')
        with open(config_path, 'w', encoding='utf8') as f:
            json.dump(dummy_config, f)
        conf_dict = {'object': dummy_config, 'path': config_path}
        return conf_dict

    @pytest.fixture
    def conf_paths(self):
        dummy_config = {'a': 1, 'b': 2}
        unorders = [
            '2020-02-02-13-50-15-406591', '2020-02-03-13-50-17-192458',
            '2020-01-03-13-50-17-192458', '2020-02-13-13-50-18-020840'
        ]
        dummy_config_dir = sd.joinpath('data/cp/')
        os.makedirs(dummy_config_dir, exist_ok=True)
        config_paths = []
        for i in unorders:
            config_path = dummy_config_dir.joinpath(f'dummy_conf_{i}.json')
            with open(config_path, 'w', encoding='utf8') as f:
                json.dump(dummy_config, f)
            config_paths.append(config_path)
            """
            作成の時差をつくるためにsleepを入れる
            -> 最新configを取得するメソッドのテスト用仕込み
            """
            time.sleep(1)
        return config_paths

    # remove_infoメソッド
    def test_target_keyで指定したitemがconfigから消えてたらTrue(self, conf, cmmock):
        # configの適当なkeyを削除対象にして，remove_info実行
        remove_target = list(conf['object'].keys())[:1]
        cmmock.remove_info(conf['path'], remove_target)
        # remove_info実行済みのjsonを再ロード
        with open(conf['path'], encoding='utf-8') as f:
            result_conf = json.load(f)
        # 生成したダミー設定データを削除
        os.remove(conf['path'])
        # 削除対象が正しく消えていたらTrue
        expected = conf['object']
        expected.pop(remove_target[0])
        assert result_conf == expected

    # add_infoメソッド
    def test_指定の情報がconfigに付加できていればTrue(self, conf, cmmock):
        # configに適当なitemを付加して保存する
        target_dict = {'c': 3}
        cmmock.add_info(conf['path'], target_dict)
        # 保存した結果を再ロード
        with open(conf['path'], encoding='utf-8') as f:
            result_conf = json.load(f)
        # 生成したダミー設定データを削除
        os.remove(conf['path'])
        # 期待する結果とあえばTrue
        expected = conf['object']
        expected['c'] = 3
        assert expected == result_conf

    # save_configメソッド
    def test_指定したパスにconfigが保存できていればTrue(self, conf, cmmock):
        # conf['path']とは別名の存在しないパスを指定
        target_path = conf['path']
        target_path.rename('target_conf.json')
        cmmock.save_config(conf['object'], target_path)
        if target_path.exists:
            assert True
            os.remove(target_path)
        else:
            assert False

    # load_configメソッド
    def test_指定先のconfigデータをロードできればTrue(self, conf, cmmock):
        expected_keys = conf['object'].keys()
        cmmock.load_config(conf['path'], expected_keys)
        os.remove(conf['path'])

    # get_newest_filepathメソッド
    def test_filenameモードで最新のconfigが取得できていればTrue(self, conf_paths, cmmock):
        config_dir = conf_paths[0].parent
        result_path = \
            cmmock.get_newest_filepath(config_dir, searchmode='filename')
        for cp in conf_paths:
            os.remove(cp)
        expected = conf_paths[-1]
        assert result_path == expected

    def test_ファイル作成情報から最新のconfigが取得できていればTrue(self, conf_paths, cmmock):
        config_dir = conf_paths[0].parent
        result_path = cmmock.get_newest_filepath(config_dir)
        for cp in conf_paths:
            os.remove(cp)
        expected = conf_paths[-1]
        assert result_path == expected
