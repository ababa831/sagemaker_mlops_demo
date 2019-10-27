# experiments

上位階層にあるDockerfileでbuildしたimageのコンテナ内で実行する場合は


```ホストOSがmacの場合
$ jupyter notebook --ip="0.0.0.0" --allow-root
```

```ホストOSがlinuxの場合
$ jupyter notebook --ip="*" --allow-root
```

で起動して，提示されたURLをアクセス