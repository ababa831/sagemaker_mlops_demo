import os
from pathlib import Path
import platform


class ConfigManager(object):
    def create_config(self, dst_path):
        raise NotImplementedError

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
