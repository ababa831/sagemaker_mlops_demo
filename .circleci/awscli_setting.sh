# CircleCI管理画面のContextsまたはEnvironment Variablesで設定したAWSのアクセスキー環境変数から，AWS CLIで参照する設定プロファイルを作成
mkdir ~/.aws 
echo -e "[default]\naws_access_key_id=$AWS_ACCESS_KEY_ID_DEV\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_DEV\n" > ~/.aws/credentials
echo -e "[stg]\naws_access_key_id=$AWS_ACCESS_KEY_ID_STG\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_STG\n" >> ~/.aws/credentials
echo -e "[prd]\naws_access_key_id=$AWS_ACCESS_KEY_ID_PRD\naws_secret_access_key=$AWS_SECRET_ACCESS_KEY_PRD\n" >> ~/.aws/credentials
echo -e "[default]\nregion=$REGION\n[stg]\nregion=$REGION\n[prd]\nregion=$REGION\n" > ~/.aws/config