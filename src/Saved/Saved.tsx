// src/Saved/Saved.tsx

import React, { useEffect, useState, useRef } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  InputAdornment,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';
import UploadIcon from '@mui/icons-material/Upload';
import CloseIcon from '@mui/icons-material/Close';
import LoadingScreen from '../LoadingScreen/LoadingScreen';
import './Saved.css';
import { useLanguage } from '../contexts/LanguageContext';
import { t } from '../translation';

interface SavedItem {
  name: string;
  timestamp: string;
  video: string; 
}

interface SocketMessageProps {
  result: string;
  function: string;
}

interface SavedProps {
  socketRef: React.MutableRefObject<WebSocket | null>;
  socketMessage: SocketMessageProps | null;
  isConnected: boolean;
}

const Saved: React.FC<SavedProps> = ({ socketRef, socketMessage, isConnected }) => {
  const [savedItems, setSavedItems] = useState<SavedItem[]>([]);
  const [filteredItems, setFilteredItems] = useState<SavedItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null); // Added error state
  const [selectedItem, setSelectedItem] = useState<SavedItem | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { language } = useLanguage();

  const handleDelete = (item: SavedItem) => {
    if (window.confirm(t('confirm_delete', language) || `"${item.name}" を削除してもよろしいですか？`)) {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const deleteMessage = JSON.stringify({
          function: 'delete_saved',
          kwargs: {
            name: item.name,
            timestamp: item.timestamp,
          },
        });
        socketRef.current.send(deleteMessage);
        setSavedItems((prev) =>
          prev.filter(
            (i) => !(i.name === item.name && i.timestamp === item.timestamp)
          )
        );
        setFilteredItems((prev) =>
          prev.filter(
            (i) => !(i.name === item.name && i.timestamp === item.timestamp)
          )
        );
      }
    }
  };

  const handleDownload = (item: SavedItem) => {
    const link = document.createElement('a');
    link.href = `data:video/webm;base64,${item.video}`;
    link.download = `${item.name}.webm`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const openModal = (item: SavedItem) => {
    setSelectedItem(item);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setSelectedItem(null);
    setIsModalOpen(false);
  };

  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      uploadFile(files[0]);
    }
  };

  const uploadFile = (file: File) => {
    setIsUploading(true);
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = (reader.result as string).split(',')[1];
      const timestamp = new Date().toISOString();
      const name = file.name.split('.').slice(0, -1).join('.');

      const uploadMessage = JSON.stringify({
        function: 'upload_saved',
        kwargs: {
          name,
          timestamp,
          video_data: base64data,
        },
      });

      socketRef.current?.send(uploadMessage);
      setIsUploading(false);
      fetchSavedItems();
    };
    reader.readAsDataURL(file);
  };

  const fetchSavedItems = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      const listMessage = JSON.stringify({
        function: 'list_saved',
        kwargs: {},
      });
      socketRef.current.send(listMessage);
      setIsLoading(true);
    }
  };

  useEffect(() => {
    fetchSavedItems();
  }, [isConnected]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.function === 'list_saved' && data.result) {
          if (data.result === 'END') {
            setIsLoading(false);
          } else {
            const { name, timestamp, chunk } = data.result;
            setSavedItems((prev) => [...prev, { name, timestamp, video: chunk }]);
            setFilteredItems((prev) => [...prev, { name, timestamp, video: chunk }]);
          }
        } else if (data.function === 'upload_saved_success') {
          fetchSavedItems();
        } else if (data.error) {
          setError(data.error);
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Failed to parse message in Saved component:', err);
        setError(t('error_parsing', language) || 'Failed to parse server response.');
        setIsLoading(false);
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
  }, [socketRef, language]);

  useEffect(() => {
    if (searchQuery === '') {
      setFilteredItems(savedItems);
    } else {
      setFilteredItems(
        savedItems.filter(item =>
          item.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
    }
  }, [searchQuery, savedItems]);

  return (
    <>
      {!isConnected && <LoadingScreen />}
      <Box className="saved-container">
        <Box className="header">
          <Typography variant="h4" className="saved-title">
            {t('files', language)}
          </Typography>
          <Box className="upload-section">
            <input
              type="file"
              accept="video/*"
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <Tooltip title={t('upload', language)}>
              <IconButton
                color="primary"
                onClick={handleUploadClick}
                className="upload-button"
              >
                <UploadIcon />
              </IconButton>
            </Tooltip>
            {isUploading && <CircularProgress size={24} className="upload-progress" />}
          </Box>
        </Box>
        <Box className="search-bar">
          <TextField
            placeholder={t('search_placeholder', language)}
            variant="outlined"
            size="small"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            fullWidth
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
              style: { color: 'var(--text-color)' },
            }}
            className="search-input"
          />
        </Box>
        <TableContainer component={Paper} className="table-container">
          <Table aria-label="saved signs table">
            <TableHead>
              <TableRow>
                <TableCell className="table-header-cell">{t('name', language) || 'Name'}</TableCell>
                <TableCell className="table-header-cell action-header-cell" align="right">
                  {t('action', language) || 'Action'}
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredItems.map((item, index) => (
                <TableRow key={`${item.name}-${item.timestamp}-${index}`} className="table-row">
                  <TableCell className="table-cell">
                    <Button
                      onClick={() => openModal(item)}
                      className="name-button"
                    >
                      {item.name}
                    </Button>
                  </TableCell>
                  <TableCell className="table-cell action-cell" align="right">
                    <Tooltip title={t('download', language)}>
                      <IconButton
                        aria-label={t('download', language)}
                        onClick={() => handleDownload(item)}
                        className="action-button download-button"
                      >
                        <DownloadIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title={t('delete', language)}>
                      <IconButton
                        aria-label={t('delete', language)}
                        onClick={() => handleDelete(item)}
                        className="action-button delete-button"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
              {filteredItems.length === 0 && (
                <TableRow>
                  <TableCell colSpan={2} align="center">
                    <Typography variant="body1">{t('no_saved_signs', language) || 'No saved signs found.'}</Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>

        <Dialog
          open={isModalOpen}
          onClose={closeModal}
          maxWidth="md"
          fullWidth
          aria-labelledby="video-preview-title"
        >
          <DialogTitle id="video-preview-title" className="dialog-title">
            {selectedItem?.name} - {t('preview', language)}
            <IconButton
              aria-label="close"
              onClick={closeModal}
              className="close-button"
            >
              <CloseIcon />
            </IconButton>
          </DialogTitle>
          <DialogContent dividers className="dialog-content">
            {selectedItem && (
              <video controls className="preview-video">
                <source
                  src={`data:video/webm;base64,${selectedItem.video}`}
                  type="video/webm"
                />
                {t('browser_not_support', language) || 'Your browser does not support the video tag.'}
              </video>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={closeModal} color="primary" className="close-dialog-button">
              {t('close', language)}
            </Button>
          </DialogActions>
        </Dialog>

        {error && (
          <Box mt={2}>
            <Typography variant="body1" color="error">
              {error}
            </Typography>
          </Box>
        )}
      </Box>
    </>
  );
};

export default Saved;
