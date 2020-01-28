import os
from pathlib import Path
import platform
import json

from utils import Utils


class ConfigManager(object):
    def create_config(self, dst_path):
        """trainer用の設定を新規作成する．
        基本的に，trainer.pyの設定を変えたい場合は本メソッドの
        config内を設定する．
        
        パス・URI等はここで設定する必要はない．
        trainer.pyで指定した引数に応じてあとで
        出力先の設定JSONファイルにすべて記録される
        
        Parameters
        ----------
        dst_path : str
            設定ファイルの出力先パス
        """
        config = {
            'config_path': Path(dst_path).resolve(),
            'hyper_params': {
                'random_state': 0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_jobs': -1,
                'cv': 5,
                'return_train_score': True
            }
        }
        self.save_config(config, dst_path)

    def save_config(self, config, config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f, ensure_ascii=False, encoding='utf8', indent=4)

    def load_config(self, config_path, expected_keys):
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        # 想定する例外
        if not isinstance(config, dict):
            errmsg = f'設定ファイルが{type(config)}型となっている．' \
                + 'dictでなければならない．'
            raise TypeError(errmsg)

        Utils.validate_dict(config, expected_keys)

        return config

    def get_newest_filepath(self, config_dir):
        path_config_dir = Path(config_dir)
        if not path_config_dir.exists():
            raise IOError(f'{config_dir}が存在しません．')
        config_filepaths = path_config_dir.glob('*.json')

        max_birthtime = 0
        newest_filepath = None
        for cf in config_filepaths:
            birthtime = self._creation_date(cf)
            if max_birthtime < birthtime:
                max_birthtime = birthtime
                newest_filepath = cf

        return newest_filepath

    def _creation_date(self, path_to_file):
        """
        Ref: https://bit.ly/36AcG6R
        """
        if platform.system() == 'Windows':
            return os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                return stat.st_birthtime
            except AttributeError:
                # We're probably on Linux.
                # No easy way to get creation dates here,
                # so we'll settle for when its content was last modified.
                return stat.st_mtime
