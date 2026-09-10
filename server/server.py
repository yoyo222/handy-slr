import asyncio
import websockets
import json
from model.main import Session

# デモ運用: 認証を行わず，全接続を単一のデモユーザーとして扱う．
#
# 旧実装はセッションを接続元 IP (websocket.remote_address[0]) で識別していたため，
# localhost からの複数接続が同一キー '127.0.0.1' を共有していた．React の
# StrictMode は開発時に effect を二重実行するため接続が 2 本張られ，先に閉じた
# 側の finally が共有セッションを pop した結果，残った側が次のメッセージで
# KeyError になっていた．認証を廃止し，IP をキーに使わないことで解消する．
#
# 認証を復活させる場合は login.login.LoginSession を接続ごと (IP ではなく
# websocket オブジェクト単位) に保持すること．
DEMO_USER = "TEST"
AUTH_FUNCTIONS = {"login", "signup", "onOpen"}

sessionsList = {}


def get_session(user):
    # Session は MediaPipe・埋め込みモデル・原型 DB を保持し生成コストが高いため，
    # 接続間で共有する (生成は同期処理なので二重生成の競合は起きない)．
    if user not in sessionsList:
        print(f"Creating a new session for {user}")
        sessionsList[user] = Session(user)
    return sessionsList[user]


async def handler(websocket, path):
    print("Connected from: ", websocket.remote_address)
    session = get_session(DEMO_USER)
    try:
        # 認証をスキップし，フロントを即認証済みにする
        # (Auth.tsx は onOpen の result:true で setIsAuthenticated(true) する)
        await websocket.send(json.dumps({"result": True, "function": "onOpen"}))

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
                continue

            func_name = data.get("function")
            if func_name == "logout":
                break
            if func_name in AUTH_FUNCTIONS:
                # 認証なし運用のため常に成功を返す
                await websocket.send(json.dumps({"result": True, "function": func_name}))
                continue

            async for chunk in process_message(session, message):
                await websocket.send(chunk)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"Connection closed with {websocket.remote_address[0]}")

async def process_message(session, message):
    try:
        data = json.loads(message)
        if "function" in data:
            func_name = data["function"]
            
            if func_name in session.functions:
                func = session.functions[func_name]
                kwargs = {}
                args = []
                if "kwargs" in data:
                    kwargs = data["kwargs"]
                if 'args' in data:
                    args = data['args']
                
                if func_name in session.async_functions:
                    async for chunk in func(*args, **kwargs):
                        yield json.dumps({"result": chunk, "function": func_name})
                else:
                    result = func(*args, **kwargs)
                    yield json.dumps({"result": result, "function": func_name})
            else:
                yield json.dumps({"error": "Function not found"})
        else:
            yield json.dumps({"error": "Invalid message format"})
    except json.JSONDecodeError:
        yield json.dumps({"error": "Invalid JSON"})
    

print("Server Starting")
start_server = websockets.serve(handler, "127.0.0.1", 8765, max_size=10000000)
print("Server Started Successfully")

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()