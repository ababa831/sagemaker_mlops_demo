# TODO: 学習スクリプトの実装（Trains込み，ログ出力あり）
from argparse import ArgumentParser
from pathlib import Path
import sys
import traceback

from git import Repo
from trains import Task
import pandas as pd

from exceptions import InvalidColumnsError
from preprocessing import PreProcessor

repo_abspath = Path(__file__).resolve().parents[6]
repo = Repo(repo_abspath)


def parse_arg():
    parser = ArgumentParser()

    parser.add_argument('input_uri', type=str)
    parser.add_argument('-p',
                        '--project_name',
                        type=str,
                        default=_get_project_name())
    parser.add_argument('-t',
                        '--task_name',
                        type=str,
                        default=_get_head_commit_id())
    parser.add_argument('-P', '--profile', type=str, default='default')
    parser.add_argument('-o', '--output_uri', type=str, default='./outputs')

    args = parser.parse_args()

    return args


def _get_project_name():
    repo_name = repo_abspath.name
    branch = repo.active_branch
    proj_name = f'{repo_name}_{branch.name}'
    return proj_name


def _get_head_commit_id():
    return repo.active_branch.commit.hexsha


def _validate_train_data(df):
    cols_assumed = [
        'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
    ]
    if df.columns.tolist() != cols_assumed:
        raise InvalidColumnsError('DataFrameのカラムが想定したものではない．')


def load_train_data(input_path):
    try:
        train_df = pd.read_csv(input_path)
        _validate_train_data(train_df)
    except IOError:
        traceback.print_exc()
        sys.exit('\n入力した学習データは無効であるため終了')
    except InvalidColumnsError:
        traceback.print_exc()
        sys.exit('Titanicコンペで使用される学習データでないため終了')
    else:
        return train_df


if __name__ == "__main__":
    args = parse_arg()

    task = Task.init(project_name=args.project_name,
                     task_name=args.task_name,
                     output_uri=args.output_uri)
    
    train_df = load_train_data(args.input_uri)
    
    pp = PreProcessor(label='Survived')
    train_dataset = pp.get_dataset(train_df)
    
    # m = Model()
    # m.cross_validate(dataset)  # モデル保存もする

    # ログ生成あれこれ (Trainsのloggingをうまく使う)

    # Upload to s3
