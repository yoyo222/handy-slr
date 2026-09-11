import asyncio
import websockets
import json

from login.login import LoginSession
from model.main import Session

AUTH_FUNCTIONS = {"login", "signup", "onOpen"}

# Cached per user, not per connection: building a Session loads MediaPipe, the
# model and the prototype database. Never evicted on disconnect — StrictMode
# opens two connections and evicting on the first close crashed the second.
sessionsList = {}

# Single-user localhost app, so "remember me" is process memory, not a token.
rememberedUser = None


def get_session(user):
    if user not in sessionsList:
        print(f"Creating a new session for {user}")
        sessionsList[user] = Session(user)
    return sessionsList[user]


async def handler(websocket, path):
    global rememberedUser

    print("Connected from: ", websocket.remote_address)
    login = LoginSession()
    session = None

    if rememberedUser is not None:
        login.user = rememberedUser
        login.rememberMe = True
        login.used = True
        session = get_session(rememberedUser)

    try:
        # The frontend never sends onOpen, it only listens for it.
        await websocket.send(json.dumps(
            {"result": session is not None, "function": "onOpen"}
        ))

        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON"}))
                continue

            func_name = data.get("function")

            if func_name == "logout":
                rememberedUser = None
                break

            if func_name in AUTH_FUNCTIONS:
                if func_name == "onOpen":
                    await websocket.send(json.dumps(
                        {"result": session is not None, "function": "onOpen"}
                    ))
                    continue

                args = data.get("args", [])
                kwargs = data.get("kwargs", {})
                ok = login.functions[func_name](*args, **kwargs)
                if ok:
                    session = get_session(login.user)
                    rememberedUser = login.user if login.rememberMe else None
                await websocket.send(json.dumps({"result": ok, "function": func_name}))
                continue

            if session is None:
                await websocket.send(json.dumps({"error": "Not authenticated"}))
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
