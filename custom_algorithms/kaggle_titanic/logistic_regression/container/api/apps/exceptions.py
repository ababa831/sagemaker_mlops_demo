class InvalidColumnsError(Exception):
    """DataFrameのカラム名が想定通りに揃ってない場合に発生させるエラー"""
    pass