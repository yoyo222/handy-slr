import asyncio
import websockets
import json
from datetime import datetime, timezone
from model.main import Session
from login.login import LoginSession
import uuid


loginSessionsList = {}
sessionsList = {}
namespace = uuid.NAMESPACE_DNS

async def handler(websocket, path):
    global loginSessionsList, sessionsList
    client_address = websocket.remote_address[0]
    print("Connected from: ", websocket.remote_address)
    loggingin = True
    user = None
    try:
        if client_address not in loginSessionsList:
            loginSessionsList[client_address] = LoginSession()
            print("New login session started")
        if loginSessionsList[client_address].rememberMe:
            loggingin = False
            user = loginSessionsList[client_address].user
            async for chunk in process_message(loginSessionsList[client_address], json.dumps({"function": 'onOpen'})):
                print("Auto log in")
                await websocket.send(chunk)
#make sure to fix the user session, also the login 
        async for message in websocket:
            if loggingin:
                async for chunk in process_message(loginSessionsList[client_address], message):
                    value = json.loads(chunk).values()
                    if len(value) != 2:
                        continue
                    result, function = value
                    if (function == 'login' or function == 'signup') and result:
                        loggingin = False
                        user = loginSessionsList[client_address].user
                        print("User Log in succesful")
                    else:
                        print("User Log in unsuccesful")
                    await websocket.send(chunk)
            else:
                if user not in sessionsList:
                    sessionsList[user]  = Session(user)
                    print("Creating a new session for the user")
                data = json.loads(message)
                if "function" in data and data["function"] == 'logout':
                    loginSessionsList[client_address].rememberMe = False
                    break
                async for chunk in process_message(sessionsList[user], message):
                    await websocket.send(chunk)
    finally:
        if client_address in loginSessionsList:
            if not loginSessionsList[client_address].used:
                loginSessionsList.pop(client_address)
                print("Deleted unused login session")
        print(f"Connection closed with {client_address}")

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