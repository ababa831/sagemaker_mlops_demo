# sagemaker_mlops_demo

[![CircleCI](https://circleci.com/gh/ababa893/sagemaker_mlops_demo.svg?style=shield&circle-token=0d5d72b4d4ef1239eed11095d61922cbf81d305c)](https://circleci.com/gh/ababa893/sagemaker_mlops_demo)
[![Coverage Status](https://coveralls.io/repos/github/ababa893/sagemaker_mlops_demo/badge.svg)](https://coveralls.io/github/ababa893/sagemaker_mlops_demo) 
[![Python 3.6](https://img.shields.io/badge/python-3.6.9-blue.svg)](https://www.python.org/downloads/release/python-369/)


カスタムアルゴリズムによるSageMaker API構築デモコード．
及び，そのMLOpsコード．



## Structure

```
.
├── README.md
└── custom_algorithms  # カスタムアルゴリズムAPIコード等
    └── kaggle_titanic  # Kaggleのタイタニックを題材にした分類API
        ├── logistic_regression  # ロジスティック回帰版
        │   ├── container
        │   │   └── api  # Flaskアプリケーションと関連ファイル（エントリポイント, nginx, wsgi等）
        │   │       ├── apps  # ML処理パイプライン（リソース，前処理，モデル，評価，推論等)
        │   │       │   ├── experiments  # 実験用notebook
        │   │       │   └── tests/ apps内コード単体テスト
        │   │       └── tests/ Flask アプリケーションテスト用（結合テスト）
        │   └── tests/ APIの動作確認用
        └── pyproject.toml  # Poetryのpyproject.tomlで依存関係の管理を行う
```

### NOTE

- `tests` ディレクトリはpytestによるテストコード置き場

## TODO: Intruduction

## メモ

- `git submodule update --init`
    - 最新版のpullは `git submodule foreach git pull origin master`
- `mlops_env_sample/envsettings`に.envを忘れずに設定
- CircleCIのcontextにAWSの環境変数を設定をする旨
    - AWS CLI
    - AWS VPC周りの情報（SageMaker用）
- 各種.envに環境変数を入れる
- どこでdevと本番で差分がでるか？解説を書く
- ユーザ辞書確認方法
    - `echo '登録した単語' | mecab`
- trainer.pyを実行するまえに該当バケットを作成していることを確認（IaCで設定するのが理想的）
- curl でpredictor.pyへPOST
    ```request_example
    $ curl -X POST -H 'Content-Type: application/json' -d '{"PassengerId": [1, 2, 3, 4],"Pclass": [3, 1, 3, 1],"Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley Florence Briggs Thayer", "Heikkinen, Miss. Laina", "Braund, Mr. Owen Harris"],"Sex": ["male", "female", "female", "male"],"Age": [22.0, 38.0, 26.0, 28.0],"SibSp": [1, 1, 0, 1],"Parch": [0, 0, 0, 0],"Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", , "PC 17599"],"Fare": [7.25, 71.2833, 7.925, 8.123],"Cabin": ["C85", "C85", "C85", "C85"],"Embarked": ["S", "C", "Q", "NA"]}' http://0.0.0.0:8080/invocations

    {"Survived": [1, 0, 1]}
    ```