import pickle
import re
import os
from pathlib import Path
from datetime import datetime
from git import Repo
import sagemaker
import boto3
import numpy as np
import pandas as pd
import subprocess
from argparse import ArgumentParser
from dotenv import load_dotenv

workdir = Path(__file__).resolve().parent
dotenv_path = workdir.joinpath('.env')
load_dotenv(dotenv_path)

PATH_INPUT_TRAIN = './data/train_dummy.csv'
PATH_INPUT_TEST = './data/test_dummy.csv'

endpoint_prefix = 'kaggle-logistic'
endpoint_suffix = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
limit_len_endpoint = 63  # エンドポイント名は最長63に制限される


def parser():
    """AWS環境の向き先を判定する引数を受け取る"""
    argparser = ArgumentParser()
    argparser.add_argument('-e',
                           '--env',
                           type=str,
                           default='dev',
                           dest='env',
                           help='dev, prd, stgのどれかを指定すると該当のAWSにAPIをデプロイ')
    argparser.add_argument('-a',
                           '--allow-test',
                           action='store_true',
                           dest='allow_test',
                           help='作成したエンドポイントの動作テストをする場合は' +
                           '-a --allow-testをつける')
    args = argparser.parse_args()
    return args


def get_active_branch_name():
    """アクティブなブランチ名を取得"""
    repo = Repo('.', search_parent_directories=True)
    branch = repo.active_branch
    return branch.name


def rename_endpoint(name):
    """
    エンドポイント名の修正
    英数字，ハイフン以外は全てボツ．アンダースコアはハイフンに置換
    feature, hotfix, develop, test-env等の特殊ブランチ名は消える
    """
    name_hyphened = re.sub(r'_+', '-', name)
    name_filtered = re.sub(r'test-env_', '', name_hyphened)
    name_filtered = \
        re.sub(r'feature|hotfix|develop|test-env', '', name_filtered)
    name_filtered = re.sub(r'[^a-zA-Z0-9-]+', '', name_filtered)
    if name_filtered == '':
        name_filtered = 'not-named'
    maxlen_name = \
        limit_len_endpoint - len(endpoint_prefix) - len(endpoint_suffix)
    if len(name_filtered) > maxlen_name:
        name_filtered = name_filtered[:maxlen_name]
    return name_filtered


def check_is_valid_name(name):
    """エンドポイント名のチェック"""
    # 条件設定
    expected_patter = r'[a-zA-Z0-9-]+'
    available_len_name = \
        limit_len_endpoint - len(endpoint_prefix) - len(endpoint_suffix)
    # 評価式
    eval_re = re.fullmatch(expected_patter, name)
    eval_len = available_len_name >= len(name)
    if eval_re and eval_len:
        return True
    else:
        return False


def configure_deploy_settings(args):
    env = args.env
    expected_env = ['dev', 'stg', 'prd']
    if env not in expected_env:
        raise ValueError(f'指定の環境変数は{expected_env}から選ぶ')

    # activeなブランチ名をエンドポイント名とする
    endpoint_name = get_active_branch_name()

    # 不正なエンドポイント命名が渡された場合は終了
    endpoint_name = rename_endpoint(endpoint_name)
    is_valid_name = check_is_valid_name(endpoint_name)
    err_msg = '有効なエンドポイント名ではありません．'
    err_msg += '半角英数字およびハイフンのみ有効です．'
    assert is_valid_name, err_msg

    env_uppered = env.upper()
    sagemaker_role = os.environ.get('SAGEMAKER_ROLE_' + env_uppered)
    security_group_id = os.environ.get('SECURITY_GROUP_ID_' + env_uppered)
    # subnetの数はAWS環境に応じて変える
    subnet1 = os.environ.get('SUBNET1_' + env_uppered)
    subnet2 = os.environ.get('SUBNET2_' + env_uppered)
    subnet3 = os.environ.get('SUBNET3_' + env_uppered)
    vpc_config = {
        'SecurityGroupIds': [
            security_group_id,
        ],
        'Subnets': [subnet1, subnet2, subnet3]
    }
    profile = os.environ.get('PROFILE_' + env_uppered)

    endpoint_name = \
        endpoint_prefix + endpoint_name + '-{}'.format(endpoint_suffix)

    bt3_session = boto3.Session(profile_name=profile)
    session = sagemaker.Session(boto_session=bt3_session)

    account = \
        session.boto_session.client('sts').get_caller_identity()['Account']
    role = 'arn:aws:iam::{}:role/{}'.format(account, sagemaker_role)
    region = session.boto_session.region_name
    repository_name = 'kaggle-titanic-logistic-regression-api'
    tag = read_ecr_tag()
    image = f'{account}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{tag}'

    return vpc_config, image, role, endpoint_name, session


def read_ecr_tag(filename='image_tag.txt'):
    """SageMakerで利用するECR上のイメージtagを，build_and_push時に生成したログから取得"""
    path_image_tag_info = workdir.joinpath(filename)
    with open(path_image_tag_info) as f:
        tag_name = f.readline()
    tag_name = tag_name.replace('\n', '')
    return tag_name


def create_test_data(X_input_train, X_input_test):
    """空学習用ダミーデータセット（辞書）を作成して保存する"""
    X_input = {
        'X_input_train': X_input_train,
        'X_input_test': X_input_test,
    }
    with open('./test_data.pkl', 'wb') as f:
        pickle.dump(X_input, f, protocol=2)


def fit(vpc_config, image, role, session, s3_prefix):
    # 訓練用のデータをs3に配置
    data_location = session.upload_data('./test_data.pkl',
                                        key_prefix=s3_prefix)
    subnets = vpc_config['Subnets']
    secur_group_ids = vpc_config['SecurityGroupIds']
    clf = sagemaker.estimator.Estimator(image,
                                        role,
                                        1,
                                        'ml.m4.xlarge',
                                        sagemaker_session=session,
                                        subnets=subnets,
                                        security_group_ids=secur_group_ids)
    # data_locationはコンテナのtraining_pathに配置される
    clf.fit(data_location)
    return clf


def do_test_endpoint(env):
    command = [
        'pytest', '-vv', '-s', 'tests/test_latest_endpoint.py', '-E', env
    ]
    result = subprocess.run(command)
    if result.returncode == 0:
        print('作成したAPIのテストは正常に通りました．')


def gen_endpoint_txt(endpoint_name, dst_path):
    """指定保存先にエンドポイント名を記録したtxtを生成
    
    Parameters
    ----------
    endpoint_name : str
        保存したいログを辞書でまとめたもの
    dst_path : str
        保存先ファイルパス
    """
    dst_dirpath = str(Path(dst_path).parent)
    os.makedirs(dst_dirpath, exist_ok=True)
    with open(dst_path, mode='w', encoding='utf-8') as f:
        f.write(endpoint_name)


if __name__ == '__main__':
    args = parser()
    vpc_config, image, role, endpoint_name, session = \
        configure_deploy_settings(args)

    train_df = pd.read_csv(PATH_INPUT_TRAIN)
    test_df = pd.read_csv(PATH_INPUT_TEST)
    X_input_train = np.array(train_df)
    X_input_test = np.array(test_df)
    create_test_data(X_input_train, X_input_test)

    clf = fit(vpc_config, image, role, session, endpoint_name)
    predictor = clf.deploy(1, 'ml.t2.medium', endpoint_name=endpoint_name)
    print('Created EndpointName: ' + endpoint_name)
    os.remove('./test_data.pkl')
    if parser().allow_test:
        print('作成したエンドポイントの動作テストを行います')
        do_test_endpoint(args.env)

    # エンドポイント名が記録されたtxtファイルを作成 > CircleCIとSlack通知連携に使用予定
    gen_endpoint_txt(endpoint_name, 'endpoint_name.txt')
