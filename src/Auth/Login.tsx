// src/Auth/Login.tsx

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
} from '@mui/material';
import './Login.css'; // Create a separate CSS file for Login

interface LoginProps {
  socketRef: React.MutableRefObject<WebSocket | null>;
  onLogin: (token: string) => void;
}

const Login: React.FC<LoginProps> = ({ socketRef, onLogin }) => {
  const navigate = useNavigate();
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        function: 'login',
        username,
        password,
      });
      socketRef.current.send(message);
    }
  };

  // Listen for login responses
  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      if (data.function === 'login') {
        if (data.token) {
          onLogin(data.token);
          navigate('/');
        } else if (data.error) {
          setError(data.error);
        }
      }
    };

    if (socketRef.current) {
      socketRef.current.addEventListener('message', handleMessage);
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.removeEventListener('message', handleMessage);
      }
    };
  }, [socketRef, onLogin, navigate]);

  return (
    <Box className="auth-container">
      <Paper elevation={3} className="auth-paper">
        <Typography variant="h5" component="h1" gutterBottom>
          ログイン
        </Typography>
        <form onSubmit={handleSubmit} className="auth-form">
          <TextField
            label="ユーザー名"
            variant="outlined"
            fullWidth
            margin="normal"
            required
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <TextField
            label="パスワード"
            type="password"
            variant="outlined"
            fullWidth
            margin="normal"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && (
            <Typography color="error" variant="body2" className="auth-error">
              {error}
            </Typography>
          )}
          <Button type="submit" variant="contained" color="primary" fullWidth className="auth-button">
            ログイン
          </Button>
        </form>
        <Button onClick={() => navigate('/signup')} color="secondary" fullWidth className="auth-button">
          アカウントを作成
        </Button>
      </Paper>
    </Box>
  );
};

export default Login;
