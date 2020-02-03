# sagemaker_mlops_demo

Demo codes of SageMaker API (with a customized docker image) and MLOps


## Structure

```
.
├── README.md  # This file
└── custom_algorithms  # SageMaker API codes with custom algorithms
    └── kaggle_titanic  # The "Kaggle Titanic classification" API
        ├── logistic_regression  # Using logistic regression
        │   ├── container
        │   │   └── api  # A Flask application and related setting files (entry point, nginx, wsgi, etc.)
        │   │       ├── apps  # ML pipline codes
        │   │       │   ├── experiments  # Experiment notebooks
        │   │       │   └── tests
        │   │       │       └── __init__.py
        │   │       └── tests
        │   │           └── __init__.py
        │   └── tests
        │       └── __init__.py
        └── pyproject.toml  # This file orchestrate the project and its dependencies
```

### NOTE
- `tests` contains test codes


## Intruduction
