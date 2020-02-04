# デプロイ完了後，エンドポイントが，想定した結果を返すかテストするコード
# CI/CDの自動テストに組み込む
import sys
from pathlib import Path
import json

from boto3.session import Session

# 自作モジュール追加
sd = Path(__file__).resolve().parents[1]
sys.path.append(str(sd))


def test_latest_endpoint(env_option):
    """
    最新のエンドポイントに対してテスト実行
    """
    # リクエスト先のAWS環境設定を読み込む
    from container.api.apps.config_manager import ConfigManager
    config_dir = sd.joinpath('container/api/apps/config_outputs')
    cm = ConfigManager()
    config_path = cm.get_newest_filepath(config_dir)
    no_use = []
    config = cm.load_config(config_path, no_use)
    aws_profile = config['s3_config']['aws_profile']

    # AWS環境の選択
    print('テスト対象のAWS 環境: ', env_option)
    errmsg = 'デプロイ先の環境が間違っています．'
    if env_option == 'dev':
        assert aws_profile == 'default', errmsg
    else:
        assert aws_profile == env_option, errmsg
    session = Session(profile_name=aws_profile)
    runtime_client = session.client('runtime.sagemaker',
                                    region_name='ap-northeast-1')

    # 準備
    expected_type = 'application/json'
    # テストするデータを用意する
    name_json = 'data/test_input.json'
    path_json = Path(__file__).parent.joinpath(name_json).resolve()
    with open(str(path_json)) as target:
        test_data = json.load(target)
    # テスト対象のエンドポイント名を取得
    endpoint_key = 'kaggle-logistic'
    endpoint_name = select_latest_endpoint(session, endpoint_key)
    print('\nテスト対象のエンドポイント名: ', endpoint_name)

    test_serialized = json.dumps(test_data).encode('utf-8')
    print('\nリクエストBody: \n', test_data, '\n')
    # データの送信
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',  # リクエストのMIMEタイプ
        Body=test_serialized,  # 送るデータの中身
        Accept='application/json'  # レスポンスのMIMEタイプ
    )
    # レスポンスを得る
    response_type = response['ContentType']
    result = response['Body'].read().decode('utf-8')
    result_body = json.loads(result)
    payload = response['ResponseMetadata']['HTTPHeaders']['content-length']
    print('Payload size: ', payload)
    # 推論結果の表示
    print('レスポンスbody: \n', result_body)

    # テストケース
    assert expected_type == response_type, 'レスポンスのMIMEタイプが不一致'
    assert isinstance(result_body, dict), 'レスポンス形式が正しくありません'
    expected_label = 'Survived'
    assert expected_label in result_body.keys(), 'レスポンスに目的変数のキーがない'
    expected_len = len(test_data['PassengerId'])  # Lengthがわかるkeyであれば何でも良い
    result_len = len(result_body[expected_label])
    assert expected_len == result_len, 'サンプル数に対する推論数が合わない'


def select_latest_endpoint(session, name_contains):
    client = session.client('sagemaker', region_name='ap-northeast-1')
    endpoints = client.list_endpoints(SortBy='CreationTime',
                                      StatusEquals='InService',
                                      NameContains=name_contains)
    latest_endpoint_name = endpoints['Endpoints'][0]['EndpointName']
    return latest_endpoint_name
