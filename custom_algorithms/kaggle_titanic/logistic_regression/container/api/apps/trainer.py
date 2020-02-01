# TODO: 学習スクリプトの実装（Trains込み，ログ出力あり）
from argparse import ArgumentParser
from pathlib import Path
import sys
import shutil
import traceback
import logging
from logging import getLogger, StreamHandler, Formatter

from git import Repo
import pandas as pd

from exceptions import InvalidColumnsError
from preprocessing import PreProcessor
from model import Model
from config_manager import ConfigManager
from s3_updown import S3UpDown

repo_root = 6
repo_abspath = Path(__file__).resolve().parents[repo_root]
active_branch = Repo(repo_abspath).active_branch


def parse_arg():
    parser = ArgumentParser()

    parser.add_argument('input_path', type=str)
    parser.add_argument('config_path', type=str)
    parser.add_argument('-p',
                        '--project_name',
                        type=str,
                        default=active_branch.name.replace('/', '_'))
    parser.add_argument('-t',
                        '--task_name',
                        type=str,
                        default=active_branch.commit.hexsha)
    parser.add_argument('-P', '--profile', type=str, default='dev')
    parser.add_argument('-o', '--output_s3bucket', type=str, default='sample')

    args = parser.parse_args()

    return args


def init_logger():
    logger = getLogger('trainer_sample')

    # loggerのログレベル設定
    logger.setLevel(logging.DEBUG)

    stream_handler = StreamHandler()
    # handlerのエラーメッセージのレベル
    stream_handler.setLevel(logging.DEBUG)
    # ログ出力フォーマット設定
    handler_format = \
        Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)

    # loggerにhandlerをセット
    logger.addHandler(stream_handler)

    return logger


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
    logger = init_logger()

    args = parse_arg()

    model_dir = './.models'
    child_dir = f'{args.project_name}_{args.task_name}'

    logger.info('実験設定')
    cm = ConfigManager()
    cm.create_config(args.config_path)
    s3_dst_info = {
        's3_config': {
            'aws_profile': args.profile,
            'bucket_name': args.output_s3bucket,
            'path_s3_dst': child_dir
        }
    }
    cm.add_info(args.config_path, s3_dst_info)

    logger.info('学習データのロード')
    train_df = load_train_data(args.input_path)

    logger.info('前処理・特徴量エンジニアリング')
    pp = PreProcessor(config_path=args.config_path,
                      mode='train',
                      label='Survived')
    train_dataset = pp.get_dataset(train_df)
    pp.save_transformers(child_dir=child_dir,
                         transformers_name='sample_transformers.pkl.cmp')

    logger.info('学習')
    m = Model(config_path=args.config_path, mode='train')
    m.init_model()
    m.train_with_cv(train_dataset)
    m.save_model(dst_dir=model_dir,
                 child_dir=child_dir,
                 model_name='sample_model.pkl.cmp')

    logger.info('推論時に利用する各種ファイルをS3にUpload')
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
    s3ud = S3UpDown(profile=s3_dst_info['s3_config']['aws_profile'])
    s3ud.upload(s3_src_dir_w_child, s3_dst_info['s3_config']['bucket_name'],
                s3_dst_info['s3_config']['path_s3_dst'])
