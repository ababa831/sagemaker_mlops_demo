# sagemaker_mlops_demo

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
        │   │       │   └── tests
        │   │       │       └── __init__.py
        │   │       └── tests
        │   │           └── __init__.py
        │   └── tests
        │       └── __init__.py
        └── pyproject.toml  # Poetryのpyproject.tomlで依存関係の管理を行う
```

### NOTE

- `tests` ディレクトリはpytestによるテストコード置き場

## TODO: Intruduction
