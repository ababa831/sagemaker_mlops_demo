import os
from pathlib import Path, PosixPath
import platform
import json
import sys
import subprocess

from git import Repo

from utils import Utils


class ConfigManager(object):
    def __init__(self):
        repo_abspath = Path(__file__).resolve().parents[6]
        self.repo = Repo(repo_abspath)

    def create_config(self, dst_path):
        """trainer用の設定を新規作成する．
        
        パス・URI等はここで設定する必要はない．
        trainer.pyで指定した引数に応じてあとで
        出力先の設定JSONファイルにすべて記録される
        
        Parameters
        ----------
        dst_path : str
            設定ファイルの出力先パス
        """
        pip_freezed = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
        packages = pip_freezed.stdout.decode('utf-8').split()
        config = {
            'config_path': Path(dst_path).resolve(),  # training時のみ使用
            'hyper_params': {
                'random_state': 0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_jobs': -1,
                'cv': 5,
                'return_train_score': True
            },
            'python': {
                'interpreter': sys.executable,
                'version': platform.python_version(),
                'packages': packages
            },
            'repository': {
                'active_branch': self.repo.active_branch.name,
                'commit_version': self.repo.active_branch.commit.hexsha
            }
        }
        self.save_config(config, dst_path)

    def remove_info(self, config_path, target_keys):
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        for k in target_keys:
            if k in config:
                config.pop(k)

        self.save_config(config, config_path)

    def add_info(self, config_path, target_dict):
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        for k, v in target_dict.items():
            config[k] = v

        self.save_config(config, config_path)

    def save_config(self, config, config_path):
        config = self._posixpath2str(config)
        if isinstance(config_path, PosixPath):
            os.makedirs(config_path.parent, exist_ok=True)
        else:
            os.makedirs(Path(config_path).parent, exist_ok=True)
        with open(config_path, 'w', encoding='utf8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def _posixpath2str(self, target_dict):
        if isinstance(target_dict, dict):
            for k, v in target_dict.items():
                if isinstance(v, dict):
                    target_dict[k] = self._posixpath2str(v)
                if isinstance(v, list):
                    target_dict[k] = \
                        [self._posixpath2str(v_val) for v_val in v]
                elif isinstance(v, PosixPath):
                    target_dict[k] = str(v)
        return target_dict

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
