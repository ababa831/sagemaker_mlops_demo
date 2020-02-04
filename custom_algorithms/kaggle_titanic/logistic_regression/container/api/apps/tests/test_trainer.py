from pathlib import Path
import sys
import traceback

import pandas as pd

from exceptions import InvalidColumnsError

sd = Path(__file__).parents[1].resolve()
sys.path.append(sd)


class TestTrainer(object):
    # _validate_train_dataメソッド
    def test_入力DataFrameのカラムが想定カラム通りであればTrue(self, do):
        from trainer import _validate_train_data
        _validate_train_data(do.dummy_valid_df)

    def test_入力DataFrameのカラムが欠けていればInvalidColumnsError(self, do):
        from trainer import _validate_train_data
        invalid_df = do.dummy_valid_df.drop('PassengerId', axis=1)
        try:
            _validate_train_data(invalid_df)
        except InvalidColumnsError:
            traceback.print_exc()
            assert True
        else:
            assert False

    # load_train_dataメソッド
    def test_入力データがDataFrameとして読み込めればTrue(self, do):
        from trainer import load_train_data
        result_df = load_train_data(do.data_path)
        assert isinstance(result_df, pd.core.frame.DataFrame)
