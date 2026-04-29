import json
import bcrypt


class LoginSession:
    def __init__(self):
        self.users = self.load_users()
        self.rememberMe = False
        self.user = None
        self.used = False
        self.functions = {
            "login": self.login,
            "signup": self.signup,
            'onOpen': self.onOpen,
        }
        self.async_functions = {}


    def onOpen(self):
        return self.rememberMe
    

    def load_users(self):
        try:
            with open("server/login/users.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}

    def save_users(self):
        with open("server/login/users.json", "w") as file:
            json.dump(self.users, file)

    def login(self, username, password, rememberMe):
        print("Username", username, "is attemping to log in.")
        if username in self.users:
            hashed_password = self.users[username]
            if bcrypt.checkpw(password.encode(), hashed_password.encode()):
                self.rememberMe = rememberMe
                self.used = True
                self.user = username
                return True
        return False

    def signup(self, username, password, rememberMe):
        print("Username", username, "is attemping to sign up.")
        if username in self.users:
            return False  # Username already exists
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        if username in self.users:
            return False
        self.users[username] = hashed_password
        self.save_users()
        self.rememberMe = rememberMe
        self.used = True
        self.user = username
        return True