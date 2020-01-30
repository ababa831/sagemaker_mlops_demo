# TODO: 学習スクリプトの実装（Trains込み，ログ出力あり）
from argparse import ArgumentParser
from pathlib import Path
import sys
import shutil
# import traceback

from git import Repo
# from trains import Task
import pandas as pd

from exceptions import InvalidColumnsError
from preprocessing import PreProcessor
from model import Model
from config_manager import ConfigManager
from s3_updown import S3UpDown

repo_abspath = Path(__file__).resolve().parents[6]
repo = Repo(repo_abspath)


def parse_arg():
    parser = ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('config_path', type=str)
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
        # traceback.print_exc()
        sys.exit('\n入力した学習データは無効であるため終了')
    except InvalidColumnsError:
        # traceback.print_exc()
        sys.exit('Titanicコンペで使用される学習データでないため終了')
    else:
        return train_df


if __name__ == "__main__":
    args = parse_arg()

    model_dir = './.models'
    child_dir = f'{args.project_name}_{args.task_name}'

    cm = ConfigManager()
    cm.create_config(args.config_path)
    s3_dst_info = {
        's3_config': {
            'aws_profile': args.profile,
            'bucket_name': 'mlops_samples',
            'path_s3_dst': child_dir
        }
    }
    cm.add_info(args.config_path, s3_dst_info)

    train_df = load_train_data(args.input_path)

    pp = PreProcessor(config_path=args.config_path,
                      mode='train',
                      label='Survived')
    train_dataset = pp.get_dataset(train_df)
    pp.save_transformers(child_dir=child_dir,
                         transformers_name='sample_transformers.pkl.cmp')

    m = Model(config_path=args.config_path, mode='train')
    m.init_model()
    m.train_with_cv(train_dataset)
    m.save_model(dst_dir=model_dir,
                 child_dir=child_dir,
                 model_name='sample_model.pkl.cmp')

    # S3のUpload対象にconfigと学習データも含めたいとき，以下の処理を行う
    # 1. configと学習データの情報を更新
    newinfo = {
        'config_name': Path(args.config_path).name,
        'input_name': Path(args.input_path).name
    }
    cm.add_info(args.config_path, newinfo)
    # 2. S3のUploadソースにconfigと学習データをコピー
    s3_src_dir_w_child = str(Path(model_dir).joinpath(child_dir))
    shutil.copy(args.config_path, s3_src_dir_w_child)
    shutil.copy(args.input_path, s3_src_dir_w_child)

    # S3に推論時に利用する各種ファイルをUploadする
    s3ud = S3UpDown(profile='default')
    s3ud.upload(s3_src_dir_w_child, s3_dst_info['s3_config']['bucket_name'],
                s3_dst_info['s3_config']['path_s3_dst'])
