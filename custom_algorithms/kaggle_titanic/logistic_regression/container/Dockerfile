# ECRからpullした検証環境のimage名
# (docker-compose.ymlの`service > ml_base`で設定したものにあたる)
FROM mlops_env_sample_ml_base:latest

MAINTAINER ababa831 <flvonlineconverter@gmail.com>

# 検証環境imageのPoetryで設定したPython依存関係をインストール．
# （image容量を削減するために，本番に使うモノのみを入れる.
# また，package数が増加した際のコンフリクト避けにも有用．
# e.g. sagemakerとawscliのコンフリクト （2019年夏頃実際にあった事例．））
RUN cd /opt/program && poetry install --no-dev

# Local環境 hostPCのSageMakerのAPIに関わるディレクトリをdocker imageにコピペ
# （CI ジョブを進めていくと，apiディレクトリ以下にコード差分（更新）が起こりうるので，
# 検証環境にあらかじめ入れるのではなく，できればここでCOPYさせたい）
COPY api /opt/program

# 既に/opt/programが作業ディレクトリであるはずだが，念の為再登録
WORKDIR /opt/program