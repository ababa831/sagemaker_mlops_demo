from pathlib import Path
import urllib.request
import json
import time


def test_api():
    # テストするデータを用意する
    name_test_json = 'data/test_input.json'
    path_test_json = Path(__file__).parent.joinpath(name_test_json).resolve()
    with open(path_test_json, 'r') as target:
        test_data = json.load(target)

    # シリアライズ
    test_serialized = json.dumps(test_data).encode('utf-8')
    print('リクエストBody: \n', test_data, '\n')

    # データの送信
    print('API起動まで暫く待機')
    time.sleep(60)  # CircleCI上のAPI起動待機時間
    url = 'http://0.0.0.0:8080/invocations'
    method = 'POST'
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    request = urllib.request.Request(url,
                                     data=test_serialized,
                                     headers=headers,
                                     method=method)

    # レスポンスを得る
    with urllib.request.urlopen(request) as response:
        payload = response.getheader('Content-Length')
        result = response.read().decode('utf-8')
        status_code = response.getcode()
    result_dic = json.loads(result)

    # 推論結果の表示
    print('ステータスコード:', status_code)
    print('レスポンスBody: \n', result_dic)
    errmsg = f'異常終了 HTTP status: {status_code}\n'
    assert status_code == 200, errmsg
    print('Payload size: ', int(payload))


if __name__ == "__main__":
    test_api()
