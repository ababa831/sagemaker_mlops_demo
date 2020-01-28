import json


class Utils(object):
    @classmethod
    def validate_dict(cls, dict_obj, expected_keys):
        """
        TODO: 他のmoduleでも利用しているので，Utilsクラスにまとめる
        """
        for exp_key in expected_keys:
            if exp_key not in dict_obj.keys():
                errmsg = f'{exp_key}が設定ファイルに含まれてない．'
                raise KeyError(errmsg)
