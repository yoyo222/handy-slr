// src/components/Settings.tsx

import React from 'react';
import './Settings.css';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { t } from '../translation';
import { FaLanguage } from 'react-icons/fa';
import { FaSun, FaMoon } from 'react-icons/fa';

const Settings: React.FC = () => {
  const { theme, setTheme } = useTheme();
  const { language, setLanguage } = useLanguage();

  const handleThemeChange = () => {
    const selectedTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(selectedTheme);
  };

  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedLang = e.target.value as 'jp' | 'en';
    setLanguage(selectedLang);
  };

  return (
    <div className="settings-container">
      <h2 className="settings-title">{t('settings', language)}</h2>

      <div className="settings-box">
        <div className="settings-option">
          <div className="icon-wrapper">
            {theme === 'light' ? <FaSun /> : <FaMoon />}
          </div>
          <div className="option-details">
            <label className="settings-label">{theme === 'dark' ? t('dark_mode', language) : t('light_mode', language)}</label>
            <div className="toggle-switch">
              <input
                type="checkbox"
                checked={theme === 'dark'}
                onChange={handleThemeChange}
                id="theme-toggle"
                aria-label={t('toggle_dark_mode', language) || 'Toggle Dark Mode'}
              />
              <label htmlFor="theme-toggle" className="slider round"></label>
            </div>
          </div>
        </div>

        <div className="settings-option">
          <div className="icon-wrapper">
            <FaLanguage />
          </div>
          <div className="option-details">
            <label className="settings-label">{t('language', language)}</label>
            <select
              value={language}
              onChange={handleLanguageChange}
              className="language-select"
              aria-label={t('select_language', language) || 'Select Language'}
            >
              <option value="en">{t('english', language)}</option>
              <option value="jp">{t('japanese', language)}</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
