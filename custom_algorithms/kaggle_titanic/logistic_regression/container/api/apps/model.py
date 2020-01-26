from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


class Model(object):
    def __init__(self, hyper_parameters=None, disable_update=False):
        if hyper_parameters:
            assert isinstance(hyper_parameters, dict)
        self.hyper_parameters = hyper_parameters
        self.disable_update = disable_update

    def init_model(self):
        if self.hyper_parameters is None:
            self.hyper_parameters = {
                'random_state': 0,
                'solver': 'lbfgs',
                'class_weight': 'balanced',
                'n_jobs': -1
            }
        try:
            clf = LogisticRegression(
                random_state=self.hyper_parameters['random_state'],
                solver=self.hyper_parameters['solver'],
                class_weight=self.hyper_parameters['class_weight'],
                n_jobs=self.hyper_parameters['n_jobs'])
        except KeyError as e:
            # self.hyper_parametersのkeyに想定したものが存在しない場合
            print('KeyError: ', e)
            print('LogisticRegressoionのデフォルトパラメータ値を代替として設定．')
            clf = LogisticRegression()
        finally:
            return clf

    def load_model(self, parameter_list, config_path):
        raise NotImplementedError

    def train_with_cv(self, dataset, cv=5, return_train_score=True):
        # hyper_parametersに指定されている場合はそちらを優先
        if cv in self.hyper_parameters and self.disable_update:
            cv = self.hyper_parameters['cv']
        if return_train_score in self.hyper_parameters and self.disable_update:
            return_train_score = \
                self.hyper_parameters['return_train_score']

        # TODO: 学習メソッドの続きを書く．同時に単体テスト書いてCircleCI回す
        raise NotImplementedError
