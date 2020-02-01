from pathlib import Path
import sys
import joblib
import traceback

import pandas as pd

sd = Path(__file__).parent.resolve()
sys.path.append(str(sd))


class ScoringService(object):
    from config_manager import ConfigManager
    cm = ConfigManager()
    # 推論に必要なオブジェクトは以下のディレクトリに配置(SageMakerの仕様)
    path_dir_objects = Path('/opt/ml/model')
    config_path = cm.get_newest_filepath(path_dir_objects)
    expected_keys = ['model_path', 'transformers_path']
    config = cm.load_config(config_path, expected_keys)

    model = None
    transformers = None

    @classmethod
    def check_if_all_models_exist(cls):
        """推論に仕様するモデル等が全て正常にロードできるかチェック
        
        Returns
        -------
        bool
            正常にモデルがロードできればTrueを返す
        """
        try:
            cls.load_models()
        except Exception:
            traceback.print_exc()
            print('推論用モデルロード時にエラー発生 > 404')
            return False
        else:
            print('推論用モデルが正常にロード > 200')
            return True

    @classmethod
    def load_models(cls):
        """モデルのロード．主にpingチェックで利用
        
        TODO: 辞書の設計が甘いので修正
            推論用にパスを書き換える処理が余計
            configのkeyにmodel_nameを予め実装しておけばこの余計なコードが減る
        """
        if not cls.model:
            model_name = Path(cls.config['model_path']).name
            model_path = cls.path_dir_objects.joinpath(model_name)
            cls.model = joblib.load(model_path)
            cls.config['model_path'] = model_path
        if not cls.transformers:
            trans_name = Path(cls.config['transformers_path']).name
            transformers_path = cls.path_dir_objects.joinpath(trans_name)
            cls.transformers = joblib.load(transformers_path)
            cls.config['transformers_path'] = transformers_path

    @classmethod
    def do_inference(cls, request_body):
        """与えられたdictを取り出して，
        apps内の各種MLパイプランに乗せて，最終推論結果を返す
        
        Parameters
        ----------
        request_body : dict
            リクエストbody
        
        Returns
        -------
        dict
            レスポンスbody
        """
        label = 'Survived'

        from utils import Utils
        logger = Utils.init_logger('predicton_sample')

        logger.info('前処理・特徴量エンジニアリング')
        from preprocessing import PreProcessor
        pred_df = pd.DataFrame(request_body)
        pp = PreProcessor(config_path=cls.config_path,
                          mode='pred',
                          label=label)
        pred_dataset = pp.get_dataset(pred_df)

        logger.info('推論')
        cls.load_models()
        from model import Model
        m = Model(config_path=cls.config_path, mode='pred')
        result = m.predict(pred_dataset)

        response_body = {label: result['y']}

        return response_body
