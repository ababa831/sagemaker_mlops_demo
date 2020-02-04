#!/usr/bin/env bash

# ECRからpullする検証環境のイメージ名
src_image=mlops_env_sample_ml_base

# 作成するリポジトリ名を設定
sagemaker_image=kaggle-titanic-logistic-regression-api


# [引数の設定]
# 本スクリプト実行時 引数によってSageMaker用imageのpush先（AWS環境）が変わる．
# dev/stg/prd 環境を想定
base_profile="default"
profile_to_push=${base_profile}
env=$1
if [ "$env" == "prd" ]
then
    profile_to_push="prd"
elif [ "$env" == "stg" ]
then
    profile_to_push="stg"
fi

# 使用するソースimage（ai_environment_jupyter）のタグ名を引数から受け取る．（何もなければlatest）
tag=$2
if [ "$tag" == "" ]
then
    tag="latest"
fi


# [ECR操作用関数の定義]
# TODO: 関数の引数番号の与え方が正しいかチェック
function get_account () {
    account=$(aws sts get-caller-identity --profile $1 --query Account --output text)
    if [ $? -ne 0 ]
    then
        exit 255
    fi
    echo "$account"
}
function get_region () {
    region=$(aws configure --profile $1 get region)
    region=${region:-ap-northeast-1}
    echo "$region"
}


# [AWS Develop環境のECRからベースとなるDocker imageをPull]
# アカウント取得
account=$(get_account ${base_profile}) 

# Region取得
region=$(get_region ${base_profile}) 

# ログイン
$(aws ecr get-login --profile ${base_profile} --region ${region} --no-include-email)

# ECRからpull
src_fullname=${account}.dkr.ecr.${region}.amazonaws.com/${src_image}:${tag}
docker pull ${src_fullname}
docker tag ${src_fullname} ${src_image}:${tag}


# [Push先に関する設定]
# Push先アカウント取得
account=$(get_account ${profile_to_push}) 

# Push先のRegion取得
region=$(get_region ${profile_to_push})

# ログイン
$(aws ecr get-login --profile ${profile_to_push} --region ${region} --no-include-email)

# Push先にリポジトリが存在してなかったら作成
aws ecr describe-repositories --profile ${profile_to_push} --repository-names "${sagemaker_image}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --profile ${profile_to_push} --repository-name "${sagemaker_image}" > /dev/null
fi

# Push先イメージのタグ名（日時）を取得
# ==============================================================================================
# SageMakerの設定からAPIを復元する際の注意点:
# SageMaker用imageのtagをlatestではなく一意にしておかないと，あとでimageが上書きされた際に完全復元が大変になる
# (過去のブランチを遡ってdocker imageから作り直す必要あり)
# ==============================================================================================
datetime=$(date +%Y-%m-%d-%H-%M-%S-%3N)
echo ${datetime} > ../image_tag.txt


# [SageMaker用imageをpush]
# SageMaker実行に必要なスクリプト権限を付与
chmod +x api/train
chmod +x api/serve

# ローカルAPIテスト用エントリポイントに権限を付与
chmod +x entrypoint.sh

# apiディレクトリ以下を反映させたdocker imageをビルドして，ECRヘPush
docker build  -t ${sagemaker_image} .
sagemaker_fullname="${account}.dkr.ecr.${region}.amazonaws.com/${sagemaker_image}:${datetime}"
docker tag ${sagemaker_image} ${sagemaker_fullname}
docker push ${sagemaker_fullname}