from pathlib import Path
import sys
import traceback

import pytest

sd = Path(__file__).parents[1].resolve()
sys.path.append(sd)


class TestUtils(object):
    @pytest.fixture
    def utils_(self):
        from utils import Utils
        return Utils

    def test_辞書に想定したキーが含まれている場合は正常終了(self, utils_):
        dict_obj = {'a': 1, 'b': 2}
        utils_.validate_dict(dict_obj, dict_obj.keys())

    def test_辞書に想定したキーが含まれていない場合KeyError(self, utils_):
        dict_obj = {'a': 1, 'b': 2}
        invalid_keys = ['c']
        try:
            utils_.validate_dict(dict_obj, invalid_keys)
        except KeyError:
            traceback.print_exc()
            assert True
        else:
            assert False
