from argparse import ArgumentParser
from pathlib import Path
import sys
import shutil
import traceback

from git import Repo
import pandas as pd

from exceptions import InvalidColumnsError
from preprocessing import PreProcessor
from model import Model
from config_manager import ConfigManager
from s3_updown import S3UpDown
from utils import Utils

repo_root = 6
repo_abspath = Path(__file__).resolve().parents[repo_root]
active_branch = Repo(repo_abspath).active_branch


def parse_arg():
    parser = ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('config_name', type=str)
    parser.add_argument('-p',
                        '--project_name',
                        type=str,
                        default=active_branch.name.replace('/', '_'))
    parser.add_argument('-t',
                        '--task_name',
                        type=str,
                        default=active_branch.commit.hexsha)
    parser.add_argument('-P', '--profile', type=str, default='dev')
    parser.add_argument('-o',
                        '--output_s3bucket',
                        type=str,
                        default='sample-titanic-logistic-regression')

    args = parser.parse_args()

    return args


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
    except UnicodeDecodeError:
        traceback.print_exc()
        sys.exit('csv, tsv形式でないファイルを読み込もうとしたので終了')
    else:
        return train_df


if __name__ == "__main__":
    args = parse_arg()

    logger = Utils.init_logger(args.task_name)

    model_dir = './.models'
    child_dir = f'{args.project_name}_{args.task_name}'
    working_dir = Path(__file__).parent.resolve()

    logger.info('実験設定')
    cm = ConfigManager()
    # configの保存先ディレクトリは，推論側の都合上固定している．
    config_path = working_dir.joinpath('config_outputs', args.config_name)
    cm.create_config(config_path)
    s3_dst_info = {
        's3_config': {
            'aws_profile': args.profile,
            'bucket_name': args.output_s3bucket,
            'path_s3_dst': child_dir
        }
    }
    cm.add_info(config_path, s3_dst_info)

    logger.info('学習データのロード')
    train_df = load_train_data(args.input_path)

    logger.info('前処理・特徴量エンジニアリング')
    pp = PreProcessor(config_path=config_path,
                      mode='train',
                      label='Survived')
    train_dataset = pp.get_dataset(train_df)
    pp.save_transformers(child_dir=child_dir,
                         transformers_name='sample_transformers.pkl.cmp')

    logger.info('学習')
    m = Model(config_path=config_path, mode='train')
    m.init_model()
    m.train_with_cv(train_dataset)
    m.save_model(dst_dir=model_dir,
                 child_dir=child_dir,
                 model_name='sample_model.pkl.cmp')

    logger.info('推論時に利用する各種ファイルをS3にUpload')
    # S3のUpload対象にconfigと学習データも含めたいとき，以下の処理を行う
    # 1. configと学習データの情報を更新
    newinfo = {
        'config_name': Path(config_path).name,
        'input_name': Path(args.input_path).name
    }
    cm.add_info(config_path, newinfo)
    # 2. S3のUploadソースにconfigと学習データをコピー
    s3_src_dir_w_child = str(Path(model_dir).joinpath(child_dir))
    shutil.copy(config_path, s3_src_dir_w_child)
    shutil.copy(args.input_path, s3_src_dir_w_child)

    # S3に推論時に利用する各種ファイルをUploadする
    s3ud = S3UpDown(profile=s3_dst_info['s3_config']['aws_profile'])
    s3ud.upload(s3_src_dir_w_child, s3_dst_info['s3_config']['bucket_name'],
                s3_dst_info['s3_config']['path_s3_dst'])
