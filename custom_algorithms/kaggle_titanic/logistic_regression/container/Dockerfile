FROM python:3.6

# シェルはbashにする
SHELL ["/bin/bash", "-c"]

WORKDIR /opt/program
COPY pyproject.toml .

# APIモック用とJupyter用
EXPOSE 5000 8888

# Poetryの導入
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
ENV PATH $PATH:/root/.poetry/bin

# Poetry仮想環境ディレクトリを固定化（この場合，カレントディレクトリに仮想環境が作成される）
# APIで仮想環境先のPythonインタプリタを使うので，仮想環境パスがランダムに生成されないようにしたかった
RUN mkdir -p /root/.config/pypoetry/ \
    && touch /root/.config/pypoetry/config.toml \
    && poetry config settings.virtualenvs.create true \
    && poetry config settings.virtualenvs.in-project true \
    && poetry install
    # && poetry install --no-dev

CMD ["/bin/bash"]