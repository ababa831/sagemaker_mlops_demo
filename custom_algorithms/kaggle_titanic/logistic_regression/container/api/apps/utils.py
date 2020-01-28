import json


class Utils(object):
    @classmethod
    def load_config(cls, config_path, expected_keys):
        with open(config_path, encoding='utf-8') as f:
            config = json.load(f)

        # 想定する例外
        if not isinstance(config, dict):
            errmsg = f'設定ファイルが{type(config)}型となっている．' \
                + 'dictでなければならない．'
            raise TypeError(errmsg)
        for exp_key in expected_keys:
            if exp_key not in config.keys():
                errmsg = f'{exp_key}が設定ファイルに含まれてない．'
                raise KeyError(errmsg)

        return config

    @classmethod
    def validate_dict(cls, dict_obj, expected_keys):
        """
        TODO: 他のmoduleでも利用しているので，Utilsクラスにまとめる
        """
        for exp_key in expected_keys:
            if exp_key not in dict_obj.keys():
                errmsg = f'{exp_key}が設定ファイルに含まれてない．'
                raise KeyError(errmsg)
