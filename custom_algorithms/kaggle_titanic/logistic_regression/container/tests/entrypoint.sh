#!/bin/bash

# build_and_push.shでSageMakerAPI用コンテナイメージを作成したあとのローカルテスト環境構築に使用
# このファイルのあるtestsディレクトリに移動 -> chmod +x entrypoint.shしたあとに
# `$ docker run -d -v $PWD:/tmp/working -w /tmp/working -p 8080:8080 -e {環境変数設定} ... {作成したイメージ} entrypoint.sh`
# サーバが立ち上がったあとに，
# http://localhost:8080 宛にリクエストを投げるテストコードtest_predictor.py を実行

# AWS CLI 用の環境変数を設定（docker run 時に，--envオプションで必要な環境変数が渡されていることを想定）
# （CircleCI上でdocker run する場合は，予め管理画面にてAWS CLI 用環境変数をcontextまたはEnvironment Variablesに設定しておくこと．）
mkdir ~/.aws 
echo -e "[default]\naws_access_key_id=$AWS_ACCESS_KEY_ID_DEV\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_DEV\n" > ~/.aws/credentials
echo -e "[stg]\naws_access_key_id=$AWS_ACCESS_KEY_ID_STG\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_STG\n" >> ~/.aws/credentials
echo -e "[prd]\naws_access_key_id=$AWS_ACCESS_KEY_ID_PRD\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_PRD\n" >> ~/.aws/credentials
echo -e "[default]\nregion=$REGION\n[stg]\nregion=$REGION\n[prd]\nregion=$REGION\n" > ~/.aws/config

# SageMakerAPI用 スクリプトがあるディレクトリに移動
cd /opt/program
# trainスクリプトでS3から/opt/ml/modelディレクトリに推論用ファイルをDL
train
# serveスクリプトでサーバ立ち上げ Listen状態にする
serve