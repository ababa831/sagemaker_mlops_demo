# sagemaker_mlops_demo

[![CircleCI](https://circleci.com/gh/ababa893/sagemaker_mlops_demo.svg?style=shield&circle-token=0d5d72b4d4ef1239eed11095d61922cbf81d305c)](https://circleci.com/gh/ababa893/sagemaker_mlops_demo)
[![Coverage Status](https://coveralls.io/repos/github/ababa893/sagemaker_mlops_demo/badge.svg?branch=feature/basic_api_implementation&t=blIEmn)](https://coveralls.io/github/ababa893/sagemaker_mlops_demo?branch=feature/basic_api_implementation) 
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
- どこでdevと本番で差分がでるか？解説を書く
- ユーザ辞書確認方法
    - `echo '登録した単語' | mecab`
- trainer.pyを実行するまえに該当バケットを作成していることを確認（IaCで設定するのが理想的）
- curl でpredictor.pyへPOST
    ```request_example
    $ docker exec -it 3e907e0460tent-Type:application/json' -d '{"PassengerId": [1, 2, 3],"Pclass": [3, 1, 3],"Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley Florence Briggs Thayer", "Heikkinen, Miss. Laina"],"Sex": ["male", "female", "female"],"Age": [22.0, 38.0, 26.0],"SibSp": [1, 1, 0],"Parch": [0, 0, 0],"Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],"Fare": [7.25, 71.2833, 7.925],"Cabin": ["C85", "C85", "C85"],"Embarked": ["S", "C", "S"]}' http://0.0.0.0:5008/invocations

    {"Survived": [1, 0, 1]}
    ```