# server/main.py

import numpy as np
import torch
from model.model import get_model
from model.classify import classify, toSentence
import os
import cv2
import mediapipe as mp
import base64
import datetime
import warnings
import shutil


warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

class Session:
    def __init__(self, id):
        self.id = id
        self.prev_left_hand = np.zeros((21, 3))
        self.prev_right_hand = np.zeros((21, 3))
        self.hand_landmarks = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(device=self.device)
        self.curr_sentence = []
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
        self.lastWord = None
        self.refreshCount = 0
        self.database = []
        self.classDescriptions = {}
        for databaseFolder in ['sampleSet', id]:
            if not os.path.isdir(f"server/database/{databaseFolder}"):
                continue
            for folder_name in os.listdir(f"server/database/{databaseFolder}"):
                folder_path = os.path.join(f"server/database/{databaseFolder}", folder_name)

                if os.path.isdir(folder_path):
                    self.classDescriptions[folder_name] = ''
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.npy'):
                            video_name = file_name[:-3] + 'webm'
                            file_path_npy = os.path.join(folder_path, file_name)
                            file_path_video = os.path.join(folder_path, video_name)
                            data = np.load(file_path_npy)
                            if len(data.shape) == 2 and data.shape[1] == 256:
                                self.database.append((folder_name, data, file_path_npy, file_path_video))
                            # print(data.shape)

                        elif file_name.endswith('.txt'):
                            with open(os.path.join(folder_path, file_name), 'r') as file:
                                content = file.read()
                                self.classDescriptions[folder_name] = content

        # print("Initialized with database of length:", len(self.database))

        self.functions = {
            'recieve': self.recieve,
            'stop_recording': self.stop_recording,
            'reset_data': self.reset_data,
            'record': self.record,
            'send_files': self.send_video_files,
            'send_descriptions' : self.send_class_descriptions,
            'delete_files': self.delete_files,
            'delete_folders': self.delete_folders,
            'update_description': self.update_class_description,
            'mouth_open': self.mouth_open,
        }
        self.async_functions = {
            'send_files', 'stop_recording'
        }


    def decode_image(self, base64_data):
        try:
            if base64_data.startswith('data:image/jpeg;base64,'):
                base64_data = base64_data.split(',')[1]

            image_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            return image
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def convert_mediapipe(self, image):
        try:
            if image is None:
                raise ValueError("Decoded image is None. Check if the base64 input is correct.")

            image_np = np.array(image)
            frame_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            results = self.hands.process(frame_rgb)
            frame_landmarks = np.zeros((42, 3))
            left_hand_detected = False
            right_hand_detected = False
            if results.multi_hand_landmarks:
                for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == 'Left':
                        left_hand_detected = True
                        for i, landmark in enumerate(hand_landmark.landmark):
                            frame_landmarks[i, :] = [landmark.x, landmark.y, landmark.z]
                        self.prev_left_hand = frame_landmarks[:21, :]
                    elif handedness.classification[0].label == 'Right':
                        right_hand_detected = True
                        for i, landmark in enumerate(hand_landmark.landmark):
                            frame_landmarks[i + 21, :] = [landmark.x, landmark.y, landmark.z]
                        self.prev_right_hand = frame_landmarks[21:, :]

            if not left_hand_detected:
                frame_landmarks[:21, :] = self.prev_left_hand
            if not right_hand_detected:
                frame_landmarks[21:, :] = self.prev_right_hand

            return frame_landmarks

        except Exception as e:
            print(f"Error in convert_mediapipe: {e}")
            return np.zeros((42, 3))

    def getEmbedding(self, landmarks):
        try:
            input_tensor = torch.tensor(np.array(landmarks), dtype=torch.float32, device=self.device).unsqueeze(0)
            output = self.model(input_tensor).squeeze(0).detach().cpu().numpy()
            return output
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros((1,256)) 

    def recieve(self, frame, mode="translate"):
        # if len(self.database) == 0:
        #     print("Database is empty! Either record new signs or use an account with recorded signs")
            
        current_landmarks = self.convert_mediapipe(self.decode_image(frame))
        self.hand_landmarks.append(current_landmarks)
        if mode == 'translate':
            if len(self.hand_landmarks) == 30:
                output = self.getEmbedding(self.hand_landmarks)
                sequence, costs = classify(output, 0.9, self.database)
                self.hand_landmarks = []
                if len(sequence) == 0:
                    self.lastWord = None
                    self.refreshCount += 1
                    if self.refreshCount == 4:
                        self.curr_sentence.clear()
                        self.refreshCount = 0

                    self.hand_landmarks = []
                    return 'No match found'

                if self.lastWord != sequence[0]:
                    self.curr_sentence.append(sequence[0])
                    self.lastWord = sequence[-1]
                    self.refreshCount = 0

                for word in sequence[1:]:
                    self.curr_sentence.append(word)

                if len(self.curr_sentence) > 15:
                    self.curr_sentence.pop(0)
                return ' '.join(self.curr_sentence)

        if mode == 'record':
            if self.mouth_open(frame):
                return True

        return False

    def reset_data(self):
        self.curr_sentence.clear()
        self.prev_left_hand = np.zeros((21, 3))
        self.prev_right_hand = np.zeros((21, 3))
        self.hand_landmarks = []
        self.lastWord = None
        self.refreshCount = 0
        # print("Reseted")

    def record(self, frame):
        return 'MOUTH_OPEN_TRUE' if self.recieve(frame, mode='record') else None

    async def stop_recording(self, name, video_data=None):
        embeddings = self.getEmbedding(self.hand_landmarks)
        folder_path = f'server/database/{self.id}/{name}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            with open(os.path.join(folder_path, 'description.txt'), 'w') as file:
                file.write("")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_npy = f'{timestamp}.npy'
        filename_video = f'{timestamp}.webm' 
        file_path_npy = os.path.join(folder_path, filename_npy)
        file_path_video = os.path.join(folder_path, filename_video)

        try:
            np.save(file_path_npy, embeddings)
            # print(f"Embeddings saved for {name} at {file_path_npy}")
            # print(f"Total timeframes {len(self.hand_landmarks)}")
        except Exception as e:
            print(f"Failed to save embeddings: {e}")

        if video_data:
            try:
                video_bytes = base64.b64decode(video_data)
                with open(file_path_video, 'wb') as f:
                    f.write(video_bytes)
                # print(f"Video saved for {name} at {file_path_video}")
            except Exception as e:
                print(f"Failed to save video: {e}")

        self.hand_landmarks = []

        self.database.append((name, embeddings, file_path_npy, file_path_video))


        with open(file_path_video, 'rb') as file:
            video_data = file.read()
            base64_data = base64.b64encode(video_data).decode('utf-8')
            chunk_size = 1024 * 256
            chunks = [base64_data[i:i+chunk_size] for i in range(0, len(base64_data), chunk_size)]
            for chunk in chunks:
                yield {
                    "folder": name,
                    "filename": timestamp,
                    "chunk": chunk
                }
            yield {
                    "folder": name,
                    "filename": timestamp,
                    "chunk": None
                }
        # print("SENT VIDEO")
    
    def mouth_open(self, frame):
        image = self.decode_image(frame)
        if image is None:
            raise ValueError("Decoded image is None. Check if the base64 input is correct.")
        
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                top_lip_landmarks = [13, 14]
                bottom_lip_landmarks = [17, 18]
                left_mouth_corner = 61
                right_mouth_corner = 291
                
                top_lip_y = np.mean([face_landmarks.landmark[i].y for i in top_lip_landmarks])
                bottom_lip_y = np.mean([face_landmarks.landmark[i].y for i in bottom_lip_landmarks])
                # print(bottom_lip_y-top_lip_y)
                return (bottom_lip_y - top_lip_y) > 0.03
 
    
    async def send_video_files(self):
        # print(f"Sending file with database of length {len(self.database)}")
        if len(self.database) != 0:
            yield len(self.database)
            saved = []
            for class_name, _, _, video_path in self.database:
                if os.path.exists(video_path):
                    if video_path.endswith('.webm'):
                        timestamp = os.path.basename(video_path)[:-5]
                        if os.path.exists(video_path):
                            saved.append({
                                "folder": class_name,
                                "filename": timestamp,
                                "video_path": video_path
                            })
            # print("starting chunks")
            for item in saved:
                video_path = item["video_path"]
                with open(video_path, 'rb') as file:
                    video_data = file.read()
                    base64_data = base64.b64encode(video_data).decode('utf-8')
                    chunk_size = 1024 * 256
                    chunks = [base64_data[i:i+chunk_size] for i in range(0, len(base64_data), chunk_size)]
                    for chunk in chunks:
                        yield {
                            "folder": item["folder"],
                            "filename": item["filename"],
                            "chunk": chunk
                        }
                yield {
                        "folder": item["folder"],
                        "filename": item["filename"],
                        "chunk": None
                    }
        else:
            yield "NONE"
        # print("Item sent")
        # print("chunks sent successfully.")

    def send_class_descriptions(self):
        # print(self.classDescriptions)
        return self.classDescriptions

    def delete_files(self, folder, files):
        directory_path = f'server/database/{self.id}/{folder}'
        toRemove = set(files)
        try:
            for filename in files:
                filename_npy = f'{filename}.npy'
                filename_video = f'{filename}.webm'
                file_path_npy = os.path.join(directory_path, filename_npy)
                file_path_video = os.path.join(directory_path, filename_video)

                if os.path.exists(file_path_npy):
                    os.remove(file_path_npy)
                    # print(f"Deleted {file_path_npy}")
                if os.path.exists(file_path_video):
                    os.remove(file_path_video)
                    # print(f"Deleted {file_path_video}")
        
        
            newdatabase = []
            for class_name, embeddings, npy_path, video_path in self.database:
                if not os.path.basename(video_path)[:-5] in toRemove:
                    newdatabase.append((class_name, embeddings, npy_path, video_path))
            self.database = newdatabase
        except Exception as e:
                return f"Error deleting files: {e}"

    def delete_folders(self, folders):
        newdatabase = []
        hash = set(folders)
        for class_name, embeddings, npy_path, video_path in self.database:
            if not class_name in hash:
                newdatabase.append((class_name, embeddings, npy_path, video_path))
        
        for class_name in folders:
            self.classDescriptions.pop(class_name)
        self.database = newdatabase

        for folder in folders:
            directory_path = f'server/database/{self.id}/{folder}'
            try:
                if os.path.exists(directory_path):
                    shutil.rmtree(directory_path)
            except Exception as e:
                return f"Error deleting folders: {e}"
            

    def update_class_description(self, folder, description):
        path = os.path.join(f"server/database/{self.id}/{folder}/description.txt")
        if os.path.exists(path):
            with open(path, 'w') as file:
                file.write(description)
                self.classDescriptions[folder] = description
