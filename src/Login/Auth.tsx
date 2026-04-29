// src/components/Auth.tsx

import React, { useState, useEffect } from "react";
import Button from "./Button";
import Input from "./Input";
import Label from "./Label";
import { FaCog } from 'react-icons/fa';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { t } from '../translation';
import "./Auth.css";

interface SocketMessageProps {
  result: string;
  function: string;
}

interface AuthProps {
  onLogin: (username: string, password: string, rememberMe: boolean) => void;
  onSignUp: (username: string, password: string, rememberMe: boolean) => void;
  setIsAuthenticated: (authentication: boolean) => void;
  socketMessage: SocketMessageProps | null;
  isConnected: boolean;
}

const Auth: React.FC<AuthProps> = ({
  onLogin,
  onSignUp,
  setIsAuthenticated,
  socketMessage,
  isConnected
}) => {
  const { theme, setTheme } = useTheme();
  const { language, setLanguage } = useLanguage();
  const [showSettings, setShowSettings] = useState(false);
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const toggleSettings = () => {
    setShowSettings(!showSettings);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isLogin) {
      onLogin(username, password, rememberMe);
    } else {
      if (password !== confirmPassword) {
        setErrorMessage(t('passwords_not_match', language));
        return;
      }
      if (username.length < 3) {
        setErrorMessage(t('username_length', language));
        return;
      }
      if (!/^[a-zA-Z0-9]+$/.test(username)) {
        setErrorMessage(t('username_invalid_chars', language));
        return;
      }
      if (password.length < 6) {
        setErrorMessage(t('password_length', language));
        return;
      }
      if (!/\d/.test(password)) {
        setErrorMessage(t('password_number', language));
        return;
      }
      if (!/[a-zA-Z]/.test(password)) {
        setErrorMessage(t('password_letter', language));
        return;
      }
      onSignUp(username, password, rememberMe);
    }
  };

  const toggleForm = () => {
    setIsLogin(!isLogin);
    setUsername("");
    setPassword("");
    setConfirmPassword("");
    setRememberMe(false);
    setErrorMessage("");
  };

  useEffect(() => {
    if (!isConnected) {
      setErrorMessage(t('server_not_connected', language));
      return;
    }
    setErrorMessage("");
    
    if (socketMessage?.function === 'onOpen' && socketMessage?.result) {
      setIsAuthenticated(true);
    } else if (socketMessage?.function === "signup") {
      if (socketMessage?.result) setIsAuthenticated(true);
      else setErrorMessage(t('username_exists', language));
    } else if (socketMessage?.function === "login") {
      if (socketMessage?.result) {
        setIsAuthenticated(true);
      } else {
        setErrorMessage(t('invalid_credentials', language));
      }
    }
  }, [socketMessage, isConnected, language]);

  return (
    <div className="auth-container" data-theme={theme}>
      <button 
        className="settings-toggle-button" 
        onClick={toggleSettings}
        aria-label={t('settings', language)}
      >
        <FaCog />
      </button>

      {showSettings && (
        <div className="settings-modal">
          <div className="settings-modal-content">
            <div className="settings-option">
              <span>{t('theme', language)}</span>
              <button 
                className="theme-toggle"
                onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
              >
                {theme === 'light' ? '🌙' : '☀️'}
              </button>
            </div>
            <div className="settings-option">
              <span>{t('language', language)}</span>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as 'en' | 'jp')}
                className="language-select"
              >
                <option value="en">{t('english', language)}</option>
                <option value="jp">{t('japanese', language)}</option>
              </select>
            </div>
          </div>
        </div>
      )}

      <form className="auth-form" onSubmit={handleSubmit}>
        <h2 className="auth-title">{t(isLogin ? 'login' : 'signup', language)}</h2>
        <div className="form-group">
          <Label htmlFor="username">{t('username', language)}</Label>
          <Input
            type="text"
            id="username"
            placeholder={t('enter_username', language)}
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <Label htmlFor="password">{t('password', language)}</Label>
          <Input
            type="password"
            id="password"
            placeholder={t('enter_password', language)}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        {!isLogin && (
          <div className="form-group">
            <Label htmlFor="confirmPassword">{t('confirm_password', language)}</Label>
            <Input
              type="password"
              id="confirmPassword"
              placeholder={t('confirm_password_placeholder', language)}
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>
        )}
        <div className="remember-me-container">
          <input
            type="checkbox"
            id="rememberMe"
            className="remember-me-checkbox"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
          />
          <label htmlFor="rememberMe" className="remember-me-label">
            {t('remember_me', language)}
          </label>
        </div>
        {errorMessage && <p className="error-message">{errorMessage}</p>}
        <Button type="submit" variant="default" fullWidth className="auth-button">
          {t(isLogin ? 'login' : 'signup', language)}
        </Button>
        <p className="toggle-form">
          {t(isLogin ? 'no_account' : 'have_account', language)}
          <span className="toggle-link" onClick={toggleForm}>
            {t(isLogin ? 'signup' : 'login', language)}
          </span>
        </p>
      </form>
    </div>
  );
};

export default Auth;