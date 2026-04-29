// src/components/Sidebar.tsx

import React, { useState } from 'react';
import {
  FaBars,
  FaTimes,
  FaHandPaper,
  FaCamera,
  FaFolder,
  FaBook,
  FaCog, 
  FaSignOutAlt, 
} from 'react-icons/fa';
import './Sidebar.css';
import { useLanguage } from '../contexts/LanguageContext';
import { t } from '../translation';
import Tooltip from '@mui/material/Tooltip';

interface SidebarProps {
  isOpen: boolean; 
  toggleSidebar: () => void; 
  setCurrentComponent: (component: string) => void;
  onLogout: () => void; 
}

const DocPopup: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { language } = useLanguage();
  
  return (
    <div className="doc-popup-overlay">
      <div className="doc-popup">
        <h2>{t('documentation_popup_title', language)}</h2>
        <p>{t('documentation_popup_description', language)}</p>
        <div className="doc-popup-buttons">
          <button onClick={onClose}>{t('close', language)}</button>
          <a
            href="https://github.com/yoyo222/Handy-Website"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t('visit_anyway', language)}
          </a>
        </div>
      </div>
    </div>
  );
};

const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar, setCurrentComponent, onLogout }) => {
  const { language } = useLanguage();
  const [showDocPopup, setShowDocPopup] = useState(false);

  return (
    <div className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <div className="logo-details">
        {isOpen && <div className="logo_name">Handy</div>}
        <div
          className="toggle-btn"
          onClick={toggleSidebar}
          aria-label={isOpen ? t('close_sidebar', language) || 'Close Sidebar' : t('open_sidebar', language) || 'Open Sidebar'}
          aria-expanded={isOpen}
        >
          {isOpen ? <FaTimes /> : <FaBars />}
        </div>
      </div>

      <ul className="nav-list">
        <li onClick={() => setCurrentComponent("dashboard")}>
          <Tooltip title={isOpen ? '' : t('translate', language)} placement="right">
            <div className="nav-item" aria-label={t('translate', language)}>
              <FaHandPaper className="icon" />
              {isOpen && <span className="links_name">{t('translate', language)}</span>}
            </div>
          </Tooltip>
        </li>

        <li onClick={() => setCurrentComponent("record")}>
          <Tooltip title={isOpen ? '' : t('record', language)} placement="right">
            <div className="nav-item" aria-label={t('record', language)}>
              <FaCamera className="icon" />
              {isOpen && <span className="links_name">{t('record', language)}</span>}
            </div>
          </Tooltip>
        </li>

        <li onClick={() => setCurrentComponent("saved")}>
          <Tooltip title={isOpen ? '' : t('files', language)} placement="right">
            <div className="nav-item" aria-label={t('files', language)}>
              <FaFolder className="icon" />
              {isOpen && <span className="links_name">{t('files', language)}</span>}
            </div>
          </Tooltip>
        </li>

        <li onClick={() => setShowDocPopup(true)}>
          <Tooltip title={isOpen ? '' : t('documentation', language)} placement="right">
            <div className="nav-item" aria-label={t('documentation', language)}>
              <FaBook className="icon" />
              {isOpen && <span className="links_name">{t('documentation', language)}</span>}
            </div>
          </Tooltip>
        </li>

        <li onClick={() => setCurrentComponent("settings")}>
          <Tooltip title={isOpen ? '' : t('settings', language)} placement="right">
            <div className="nav-item" aria-label={t('settings', language)}>
              <FaCog className="icon" />
              {isOpen && <span className="links_name">{t('settings', language)}</span>}
            </div>
          </Tooltip>
        </li>

        <li className="spacer"></li>

        <li onClick={onLogout}>
          <Tooltip title={isOpen ? '' : t('logout', language)} placement="right">
            <div className="nav-item" aria-label={t('logout', language)}>
              <FaSignOutAlt className="icon" />
              {isOpen && <span className="links_name">{t('logout', language)}</span>}
            </div>
          </Tooltip>
        </li>
      </ul>

      {showDocPopup && <DocPopup onClose={() => setShowDocPopup(false)} />}
    </div>
  );
};

export default Sidebar;