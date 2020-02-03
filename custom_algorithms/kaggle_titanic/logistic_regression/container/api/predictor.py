from __future__ import print_function

import json
import flask
"""
API用追加モジュール
- ScoringService
    MLパイプラインを実行するクラス
- NumpyEncoder
    レスポンス用dictにNumpy配列等含まれてたときに修正するクラス
"""
from apps.scoring_service import ScoringService
from apps.numpy_encoder import NumpyEncoder

app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    ヘルスチェック（モデルの読み込みチェック）
        `/ping`のタイムアウトは2秒
        → 2秒以内に学習済みモデルの読み込みが可能かを確認！

    Ref: https://amzn.to/32HX3IC
    """
    health = ScoringService.check_if_all_models_loaded()
    status = 200 if health else 404
    return flask.Response(response='\n',
                          status=status,
                          mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    推論リクエストは`/invocations`に対して行う
        `/invocations`のタイムアウトは60秒

    Ref: https://amzn.to/34MNmKC

    SageMakerでは，pollingは多分出来ないはず

    NOTE: httpステータスの例外は必要に応じて追記
    """
    data = None
    expected_mimetype = 'application/json'
    # 受信データのチェック
    if flask.request.content_type == expected_mimetype:
        data = flask.request.get_json()  # 受信したJSONデータ
    else:
        return flask.Response(
            response='This predictor only supports JSON data',
            status=415,
            mimetype='text/plain')
    # 推論
    result_dict = ScoringService.do_inference(data)
    # 推論結果をJSONとして返す
    return flask.Response(response=json.dumps(result_dict, cls=NumpyEncoder),
                          status=200,
                          mimetype='application/json')
