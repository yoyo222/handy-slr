# Handy AI

Handy AI is a proof-of-concept app created to demonstrate model described by "prototypical-DTW for few shot classification of sign language". New signs can be added, which can be identified in real-time without having to train any further.

Handy AIは新しい手話を追加することができ、追加された手話はさらなる訓練なしでリアルタイムに識別できます。
## Installation  インストール

Install all node modules and python packages while inside the project directory.
プロジェクトディレクトリ内で、すべてのNodeモジュールとPythonパッケージをインストールします。
```bash
yarn
pip install server/requirements.txt
```

## Usage 使用方法
Run the python server. It will be hosted in localhost: 8765.

You should see "Server Started Successfully" message shortly after.

Pythonサーバーを実行します。サーバーはlocalhost:8765でホストされます。

しばらくすると、「Server Started Successfully」（サーバーが正常に起動しました）というメッセージが表示されます。
```bash
python server/server.py
```
Then run the web app on a new terminal. It will be hosted in localhost:3000
その後、Webアプリを新たなターミナルで実行します。localhost:3000でホストされます。

```bash
yarn start
```
Username is TEST, password is test123.
You can also choose to create a new user. 
A newly added sign will be accesible only by the user.

ユーザー名はTEST、パスワードはtest123で使えます。
新しいユーザーを登録できます。
新しく追加された手話はユーザー個人だけが見られます。

## License
Favicon- Flaticon Basic License https://www.flaticon.com/free-icons/sign-language
[MIT](https://choosealicense.com/licenses/mit/)