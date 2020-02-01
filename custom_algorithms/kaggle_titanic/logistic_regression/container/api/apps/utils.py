import logging
from logging import getLogger, StreamHandler, Formatter


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

    @classmethod
    def init_logger(cls, logger_name):
        logger = getLogger(logger_name)

        # loggerのログレベル設定
        logger.setLevel(logging.DEBUG)

        stream_handler = StreamHandler()
        # handlerのエラーメッセージのレベル
        stream_handler.setLevel(logging.DEBUG)
        # ログ出力フォーマット設定
        handler_format = \
            Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(handler_format)

        # loggerにhandlerをセット
        logger.addHandler(stream_handler)

        return logger
