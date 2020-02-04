import sys
from pathlib import Path
import json

sd = Path(__file__).parent.resolve()
sys.path.append(sd)
sys.path.append(sd.parent)


class TestScoringService(object):
    """推論パイプラインの結合テスト
    """
    def test_do_inferenceで推論値yが入力サンプル数分反映されていればTrue(self, do):
        request_path = sd.joinpath('data', 'request_sample.json')
        with open(request_path, 'r') as f:
            request_data = json.load(f)

        from scoring_service import ScoringService
        response_body = ScoringService.do_inference(request_data)

        samples = len(request_data['PassengerId'])
        expected = (samples, )
        assert response_body['Survived'].shape == expected
