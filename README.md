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

