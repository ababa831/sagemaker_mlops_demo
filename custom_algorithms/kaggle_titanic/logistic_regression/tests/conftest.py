import pytest


def pytest_addoption(parser):
    """pytest用コマンドラインオプション引数を追加"""
    parser.addoption('-E', '--env', default='dev',
                     action='store', type=str, dest='env',
                     help='pro or dev環境かを選択するためのオプション')


@pytest.fixture
def env_option(request):
    return request.config.getoption('--env')
