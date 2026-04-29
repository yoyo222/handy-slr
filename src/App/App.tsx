// src/App/App.tsx

import React, { useEffect, useRef, useState } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Record from '../Record/Record';
import Saved from '../Saved/Saved';
import FileViewer from '../Saved/FileViewer';
import Settings from '../Settings/Settings'; 
import LoadingScreen from '../LoadingScreen/LoadingScreen';
import Auth from '../Login/Auth'; 
import './App.css';
import '../VideoStyles.css'; 
import Webcam from 'react-webcam';
import { FaPlay, FaCopy, FaQuestionCircle, FaCopy as FaCopyIcon } from 'react-icons/fa'; 
import { ThemeProvider } from '../contexts/ThemeContext'; 
import { LanguageProvider, useLanguage } from '../contexts/LanguageContext';
import { t } from '../translation'; 
import { FaEye, FaEyeSlash } from 'react-icons/fa'; 

interface SocketMessageProps {
  result: string;
  function: string;
}

interface HomeProps {
  socketRef: React.MutableRefObject<WebSocket | null>;
  socketMessage: SocketMessageProps | null;
  isConnected: boolean;
}

interface FileType {
  name: string;
  video: string;
}

interface FolderType {
  name: string;
  description: string;
  files: FileType[];
}

interface FolderData {
  [folderName: string]: {
    [filename: string]: {
      chunks: string[];
    };
  };
}

class VideoChunkProcessor {
  private folderData: FolderData = {};
  public startUpLength: number = -1;

  processChunk(socketMessage: {
    result: {
      folder: string;
      filename: string;
      chunk: string;
    };
  }): void {
    const { folder, filename, chunk } = socketMessage.result;

    if (!this.folderData[folder]) {
      this.folderData[folder] = {};
    }

    if (!this.folderData[folder][filename]) {
      this.folderData[folder][filename] = {
        chunks: [],
      };
    }

    this.folderData[folder][filename].chunks.push(chunk);
  }

  reconstructVideo(folder: string, filename: string): string {
    const fileEntry = this.folderData[folder]?.[filename];

    if (!fileEntry) {
      console.warn(`No chunks found for ${folder}/${filename}`);
      return '';
    }

    if (this.startUpLength > 0) {
      this.startUpLength -= 1;
    }

    const fullVideo = fileEntry.chunks.join('');
    return fullVideo;
  }

  isReady(): boolean {
    return this.startUpLength === 0;
  }

  /**
   * Get the entire folder data
   * @returns Complete FolderData object
   */
  getFolderData(): FolderData {
    return this.folderData;
  }
}








const Home: React.FC<HomeProps> = ({ socketRef, socketMessage, isConnected }) => {
  const [subtitleText, setSubtitleText] = useState<string>('');
  const [isVideoVisible, setIsVideoVisible] = useState(false);
  const [isOverlayVisible, setIsOverlayVisible] = useState(true); 
  const [isHelpOpen, setIsHelpOpen] = useState(false); 
  const [isCopyPopupOpen, setIsCopyPopupOpen] = useState(false); 
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showTutorial, setShowTutorial] = useState(true);
  const { language } = useLanguage(); 
  const [isTutorialFading, setIsTutorialFading] = useState(false);

  useEffect(() => {
    if (showTutorial) {
      const timer = setTimeout(() => {
        setIsTutorialFading(true);
        setTimeout(() => {
          setShowTutorial(false);
          setIsTutorialFading(false);
        }, 500); 
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, []);

  const toggleOverlay = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.stopPropagation();
    setIsOverlayVisible((prev) => !prev);
  };

  const handleClick = () => {
    setIsVideoVisible((prev) => !prev);
    const message = JSON.stringify({
      function: 'reset_data',
    });

    socketRef.current?.send(message);
    setSubtitleText('');
  };

  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvasRef.current = canvas;
  }, []);

  useEffect(() => {
    if (isVideoVisible && webcamRef.current) {
      const updateWebcamDimensions = () => {
        const webcamElement = webcamRef.current?.video;

        if (webcamElement) {
        }
      };

      updateWebcamDimensions();

      window.addEventListener('resize', updateWebcamDimensions);

      return () => {
        window.removeEventListener('resize', updateWebcamDimensions);
      };
    }
  }, [isVideoVisible]);

  useEffect(() => {
    if (isVideoVisible && webcamRef.current && canvasRef.current) {
      const captureImage = () => {
        const video = webcamRef.current?.video;
        if (!video) return;

        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');

        if (canvas && ctx) {
          const videoWidth = video.videoWidth;
          const videoHeight = video.videoHeight;

          canvas.width = videoWidth;
          canvas.height = videoHeight;

          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const base64Image = canvas.toDataURL('image/jpeg');
          if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            const message = JSON.stringify({
              function: 'recieve',
              args: [base64Image],
              kwargs: { mode: 'translate' },
            });

            socketRef.current.send(message);
          }
        }
      };

      const intervalId = setInterval(captureImage, 50);

      return () => clearInterval(intervalId);
    }
  }, [isVideoVisible, socketRef]);
  
  useEffect(() => {
    if (socketMessage && socketMessage.function === 'recieve' && socketMessage.result) {
      if (socketMessage.result === 'No match found') {
        setSubtitleText('');
      } else {
        setSubtitleText(socketMessage.result);
      }
    }
  }, [socketMessage]);

  const [webcamDimensions, setWebcamDimensions] = useState({
    width: 0,
    height: 0,
    top: 0,
    left: 0,
  });

  const toggleHelp = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.stopPropagation();
    setIsHelpOpen(true);
  };

  const closeHelp = () => {
    setIsHelpOpen(false);
  };

  const handleCopy = () => {
    if (subtitleText) {
      navigator.clipboard.writeText(subtitleText)
        .then(() => {
          setIsCopyPopupOpen(true);
          setTimeout(() => setIsCopyPopupOpen(false), 2000); 
        })
        .catch((err) => {
          console.error('Failed to copy text: ', err);
        });
    }
  };

  return (
    <div onClick={handleClick} className="video-container home-container">
      <button className="help-button" onClick={toggleHelp} aria-label={t('help', language)}>
        <FaQuestionCircle size={20} />
        {t('Help', language)}
      </button>

      <button 
        className="secondary-button" 
        onClick={handleCopy} 
        aria-label={t('copy_subtitles', language)}
      >
        <FaCopyIcon size={20} />
        {t('Copy as text', language)}
      </button>

      {isHelpOpen && (
        <div className="modal-overlay" onClick={closeHelp}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close-button" onClick={closeHelp} aria-label={t('Close Help Modal', language)}>
              &times;
            </button>

            <h2>{t('How to Use', language)}</h2>
            <p>{t('Help Instructions', language)}:</p>
            <p>{t('This app translates sign language into subtitles below your screen. Here’s how to get started:', language)}</p>
            
            <h3>{t('Start Signing', language)}</h3>
            <p>{t('Tap anywhere on the screen to begin. When the blur disappears, align your face with the outline for accurate sign recognition.', language)}</p>

            <h3>{t('Positioning for Signs', language)}</h3>
            <p>
              {t('- For two-handed signs, stay within the frame.', language)}<br />
              {t('- For one-handed signs (like the alphabet), rest your left hand on your stomach, as shown in the outline.', language)}
            </p>

            <h3>{t('Device Limitations', language)}</h3>
            <p>{t('Tracking may be slower on some devices. If subtitles aren’t appearing, try signing a bit slower.', language)}</p>

            <h3>{t('Copying Subtitles', language)}</h3>
            <p>{t('Press the "Copy" button under this help section to copy subtitles.', language)}</p>

            <h3>{t('Disabling the Outline', language)}</h3>
            <p>{t('To hide the outline, tap the button in the bottom-right corner of the screen.', language)}</p>
          </div>
        </div>
      )}

      {isCopyPopupOpen && (
        <div className="modal-overlay" onClick={() => setIsCopyPopupOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close-button" onClick={() => setIsCopyPopupOpen(false)} aria-label={t('close_copy_popup', language)}>&times;</button>
            <p>{t('Copied to your clipboard!', language)}</p>
          </div>
        </div>
      )}

      <Webcam
        audio={false}
        ref={webcamRef}
        className={`video-element ${isVideoVisible ? 'visible' : 'blurred'}`}
        style={{
          objectFit: 'cover', 
        }}
      />

      {isVideoVisible && isOverlayVisible && (
        <img
          src="/greyperson.png"
          className="overlay-image"
          alt="Overlay"
        />
      )}

      {!isVideoVisible && showTutorial && (
        <div className={`tutorial-overlay ${isTutorialFading ? 'fade-out' : ''}`}>
          <div className="tutorial-content">
            <div className="tutorial-message">
              <span className="pulse-dot"></span>
              {t('screen_paused', language)}
            </div>
            <div className="tutorial-instructions">
              <span className="arrow">↗️</span> {t('press_help_for_instructions', language)}
            </div>
            <div className="tutorial-hint">
              {t('tap_to_start_translating', language)}
            </div>
          </div>
        </div>
      )}

      
      {subtitleText && (
        <div className="subtitle-container">
          <p className="subtitle-text">{subtitleText.split(' ').slice(-15).join(' ')}</p>
        </div>
      )}
      <button 
        className="toggle-overlay-button" 
        onClick={toggleOverlay}
        aria-checked={isOverlayVisible}
        role="switch"
        aria-label={isOverlayVisible ? t('hide_overlay', language) : t('show_overlay', language)}
      >
      </button>
    </div>
  );
};









const App: React.FC = () => {
  const socketRef = useRef<WebSocket | null>(null);
  const [socketMessage, setSocketMessage] = useState<SocketMessageProps | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentComponent, setCurrentComponent] = useState("auth");
  const videoProcessorRef = useRef(new VideoChunkProcessor());
  const [folders, setFolders] = useState<FolderType[]>([]);
  const { language } = useLanguage(); 
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const toggleSidebar = () => setIsSidebarOpen(prev => !prev);

  const onAuthentication = (state: boolean) => {
    if (state) {
      setIsAuthenticated(true);
      console.log("Authenticated!");
      setCurrentComponent("dashboard");
      const requestList = JSON.stringify({
        function: 'send_descriptions',
        kwargs: {},
      });
      socketRef.current?.send(requestList);
    } else {
      setIsAuthenticated(false);
    }
  };

  const handleLogin = (username: string, password: string, rememberMe: boolean) => {
    if (!isConnected) return;
    const requestList = JSON.stringify({
      function: 'login',
      kwargs: {
        username: username,
        password: password,
        rememberMe: rememberMe,
      },
    });
    socketRef.current?.send(requestList);
  };

  const handleSignUp = (username: string, password: string, rememberMe: boolean) => {
    if (!isConnected) return;
    const requestList = JSON.stringify({
      function: 'signup',
      kwargs: {
        username: username,
        password: password,
        rememberMe: rememberMe,
      },
    });
    socketRef.current?.send(requestList);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setCurrentComponent("auth");
    const logoutMessage = JSON.stringify({
      function: 'logout',
      kwargs: {},
    });
    socketRef.current?.send(logoutMessage);
  };

  useEffect(() => {
    const connectWebSocket = () => {
      socketRef.current = new WebSocket('ws://localhost:8765');

      socketRef.current.onopen = () => {
        setIsConnected(true);
        console.log("WebSocket connected");
      };

      socketRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data && "result" in data && "function" in data) {
            setSocketMessage({ result: data.result, function: data.function });
          }
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      socketRef.current.onclose = () => {
        setIsConnected(false);
        console.log("WebSocket disconnected");
      };

      socketRef.current.onerror = (error) => {
        setIsConnected(false);
        console.error("WebSocket error:", error);
      };
    };

    connectWebSocket();

    const intervalId = setInterval(() => {
      if (socketRef.current) {
        if (socketRef.current.readyState !== WebSocket.OPEN) {
          console.log("WebSocket not open, attempting to reconnect...");
          connectWebSocket();
        }
      } else {
        console.log("WebSocket is null, attempting to reconnect...");
        connectWebSocket();
      }
    }, 5000);

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      clearInterval(intervalId);
    };
  }, []);

  const setUpDescriptions = (descriptions: { [folderName: string]: string } | string) => {
    const updatedFolders: FolderType[] = [];

    for (const [name, description] of Object.entries(descriptions)) {
      const existingFolderIndex = folders.findIndex(folder => folder.name === name);

      if (existingFolderIndex !== -1) {
        const existingFolder = { ...folders[existingFolderIndex], description };
        updatedFolders.push(existingFolder);
      } else {
        updatedFolders.push({ name, description, files: [] });
      }
    }

    setFolders(updatedFolders);
  };

  const updateFoldersWithVideo = (folder: string, filename: string, videoBase64: string) => {
    console.log("!");
    setFolders(prevFolders => {
      const updatedFolders = [...prevFolders];

      let folderIndex = updatedFolders.findIndex(f => f.name === folder);

      if (folderIndex === -1) {
        const newFolder: FolderType = {
          name: folder,
          description: '', 
          files: [{ name: filename, video: videoBase64 }],
        };
        updatedFolders.push(newFolder);
      } else {
        const existingFileIndex = updatedFolders[folderIndex].files.findIndex(f => f.name === filename);

        if (existingFileIndex === -1) {
          updatedFolders[folderIndex] = {
            ...updatedFolders[folderIndex],
            files: [
              ...updatedFolders[folderIndex].files,
              { name: filename, video: videoBase64 },
            ],
          };
        } else {
          updatedFolders[folderIndex].files[existingFileIndex] = {
            name: filename,
            video: videoBase64,
          };
        }
      }

      return updatedFolders;
    });
  };

  useEffect(() => {
    if (!isConnected) return;

    if (socketMessage?.function === 'send_descriptions' && socketMessage?.result) {
      setUpDescriptions(socketMessage?.result);
      console.log('Descriptions received');
      const requestList = JSON.stringify({
        function: 'send_files',
        kwargs: {},
      });
      socketRef.current?.send(requestList);
    } else if (
      (socketMessage?.function === 'send_files' || socketMessage?.function === 'stop_recording') &&
      socketMessage?.result
    ) {
      if (!isNaN(Number(socketMessage.result)) && isFinite(Number(socketMessage.result))) {
        videoProcessorRef.current.startUpLength = Number(socketMessage.result);
      } else if (socketMessage.result === "NONE") {
        setIsReady(true);
      } else {
        const { folder, filename, chunk } = socketMessage.result as unknown as {
          folder: string;
          filename: string;
          chunk: string;
        };
        console.log({ folder, filename, chunk });

        videoProcessorRef.current.processChunk({
          result: { folder, filename, chunk },
        });

        if (!chunk) {
          const video = videoProcessorRef.current.reconstructVideo(folder, filename);
          updateFoldersWithVideo(folder, filename, video);
          if (videoProcessorRef.current.isReady()) {
            setIsReady(true);
          }
        }
      }
    }
  }, [socketMessage, isConnected]);

  const renderComponent = () => {
    if (!isAuthenticated) {
      return (
        <Auth
          onLogin={handleLogin}
          onSignUp={handleSignUp}
          setIsAuthenticated={onAuthentication}
          socketMessage={socketMessage}
          isConnected={isConnected}
        />
      );
    }
    switch (currentComponent) {
      case "dashboard":
        return <Home socketRef={socketRef} socketMessage={socketMessage} isConnected={isConnected} />;
      case "record":
        return <Record socketRef={socketRef} socketMessage={socketMessage} isConnected={isConnected} />;
      case "saved":
        return <FileViewer socketRef={socketRef} socketMessage={socketMessage} isConnected={isConnected} folders={folders} setFolders={setFolders} />;
      case "settings":
        return <Settings />;
      default:
        return <Home socketRef={socketRef} socketMessage={socketMessage} isConnected={isConnected} />;
    }
  };

  return (
    <ThemeProvider>
      <LanguageProvider>
        <Router>
          <div className="app-container">
            {isAuthenticated && (
              <Sidebar 
                isOpen={isSidebarOpen} 
                toggleSidebar={toggleSidebar} 
                setCurrentComponent={setCurrentComponent} 
                onLogout={handleLogout} 
              />
            )}
            <div className={`home-section ${isSidebarOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
              {((!isConnected || !isReady) && isAuthenticated) && <LoadingScreen />}
              {renderComponent()}
            </div>
          </div>
        </Router>
      </LanguageProvider>
    </ThemeProvider>
  );
};

export default App;
