import subprocess
from pathlib import Path
import os


class S3UpDown(object):
    def __init__(self, profile='default'):
        self.profile = profile

    def upload(self, srcpath, bucket_name, s3_dst_dir):
        """AWS CLIで，指定したバケット，ディレクトリにUpload
        boto3を使うよりシンプルに書ける
        
        Parameters
        ----------
        srcpath : str
            ファイルソース
        bucket_name : str
            S3バケット名
        s3_dst_dir : str
            送り先のディレクトリパス
            NOTE: ドライブ名（s3://），バケット名は含めない
        
        Raises
        ------
        FileNotFoundError
            S3へアップロードするソースファイルが見つからない場合に発生
        """
        if not Path(srcpath).exists:
            raise FileNotFoundError(f'指定したファイル{srcpath}が存在しない')

        # 簡単な文字列処理
        # 1. s3のドライブ？名はs3 upload時コード上で与えるので，重複回避
        s3_dst_dir = s3_dst_dir.replace('s3://', '')
        # 2. バケット名と指定のS3パスに重複がある場合は取り除く
        parts_dstpath = Path(s3_dst_dir).parts
        if bucket_name == parts_dstpath[0]:
            s3_dst_dir = '/'.join(parts_dstpath[1:])
        # 3. パスの作成 (Path('hoge').joinpath で上手く処理できないので代替）
        final_dstpath = f's3://{bucket_name}/{s3_dst_dir}'

        # S3 Upload
        command = ['aws', 's3', 'cp', srcpath, final_dstpath, '--recursive']
        command += ['--profile', self.profile]
        subprocess.run(command)

    def download(self, bucket_name, s3_srcpath, dstpath):
        """学習済みモデルをDL

        Parameters
        ----------
        bucket_name : str
            学習済みモデルがあるS3バケット名
        s3_srcpath : str
            S3のバケットを除く配置場所のパス
        dstpath : str
            DL&保存先のディレクトリパス
        """
        os.makedirs(dstpath, exist_ok=True)

        # Pathlibで文字列処理が厄介であるため，文字列に代入する形を取っている
        src_in_s3 = f's3://{bucket_name}/{s3_srcpath}'

        # 再帰的にファイルをDL
        command = ['aws', 's3', 'cp', src_in_s3, dstpath, '--recursive']
        command += ['--profile', self.profile]
        subprocess.run(command)
