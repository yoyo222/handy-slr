// src/Auth/Signup.tsx

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
} from '@mui/material';
import './Signup.css'; // Create a separate CSS file for Signup

interface SignupProps {
  socketRef: React.MutableRefObject<WebSocket | null>;
}

const Signup: React.FC<SignupProps> = ({ socketRef }) => {
  const navigate = useNavigate();
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [confirmPassword, setConfirmPassword] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError("パスワードが一致しません");
      return;
    }
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        function: 'signup',
        username,
        password,
      });
      socketRef.current.send(message);
    }
  };

  // Listen for signup responses
  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      if (data.function === 'signup') {
        if (data.message) {
          setSuccess(data.message);
          setError(null);
          setTimeout(() => navigate('/login'), 2000);
        } else if (data.error) {
          setError(data.error);
          setSuccess(null);
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
  }, [socketRef, navigate]);

  return (
    <Box className="auth-container">
      <Paper elevation={3} className="auth-paper">
        <Typography variant="h5" component="h1" gutterBottom>
          サインアップ
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
          <TextField
            label="パスワード確認"
            type="password"
            variant="outlined"
            fullWidth
            margin="normal"
            required
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
          {error && (
            <Typography color="error" variant="body2" className="auth-error">
              {error}
            </Typography>
          )}
          {success && (
            <Typography color="primary" variant="body2" className="auth-success">
              {success}
            </Typography>
          )}
          <Button type="submit" variant="contained" color="primary" fullWidth className="auth-button">
            サインアップ
          </Button>
        </form>
        <Button onClick={() => navigate('/login')} color="secondary" fullWidth className="auth-button">
          ログインページへ
        </Button>
      </Paper>
    </Box>
  );
};

export default Signup;
