# server/main.py

import numpy as np
import torch
from model.model import get_model
from model.classify import classify, toSentence
from model.calibrate import auto_threshold, trim_active_span
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
        self.presence_mask = []
        self.left_missing_count = 0
        self.right_missing_count = 0
        self.both_missing_count = 0
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
                        if file_name.endswith('.npy') and not file_name.endswith('.presence.npy'):
                            video_name = file_name[:-3] + 'webm'
                            presence_name = file_name[:-4] + '.presence.npy'
                            file_path_npy = os.path.join(folder_path, file_name)
                            file_path_video = os.path.join(folder_path, video_name)
                            file_path_presence = os.path.join(folder_path, presence_name)
                            data = np.load(file_path_npy)
                            if len(data.shape) == 2 and data.shape[1] == 256:
                                if os.path.exists(file_path_presence):
                                    presence = np.load(file_path_presence)
                                else:
                                    presence = np.ones((data.shape[0], 2), dtype=np.uint8)
                                self.database.append((folder_name, data, file_path_npy, file_path_video, presence))
                            # print(data.shape)

                        elif file_name.endswith('.txt'):
                            with open(os.path.join(folder_path, file_name), 'r') as file:
                                content = file.read()
                                self.classDescriptions[folder_name] = content

        # print("Initialized with database of length:", len(self.database))

        # Calibrated decoder (validated on the continuous bench): conformal
        # auto-threshold from the registered recordings + 3-chunk debounce.
        self.DEBOUNCE = 3
        self.threshold = auto_threshold(self.database)
        print(f"[calibration] no-match threshold = {self.threshold:.3f} "
              f"({len(self.database)} prototypes)")
        self.run_label = None      # debounce state across buffers
        self.run_len = 0
        self.prev_emitted = None

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

            # Decay a frozen hand to zero after MISSING_RESET frames (~250ms) so
            # prolonged absence reads as "no hand" rather than a stale ghost pose.
            MISSING_RESET = 5
            if left_hand_detected:
                self.left_missing_count = 0
            else:
                self.left_missing_count += 1
                if self.left_missing_count > MISSING_RESET:
                    self.prev_left_hand = np.zeros((21, 3))
            if right_hand_detected:
                self.right_missing_count = 0
            else:
                self.right_missing_count += 1
                if self.right_missing_count > MISSING_RESET:
                    self.prev_right_hand = np.zeros((21, 3))

            if not left_hand_detected:
                frame_landmarks[:21, :] = self.prev_left_hand
            if not right_hand_detected:
                frame_landmarks[21:, :] = self.prev_right_hand

            return frame_landmarks, (left_hand_detected, right_hand_detected)

        except Exception as e:
            print(f"Error in convert_mediapipe: {e}")
            return np.zeros((42, 3)), (False, False)

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

        current_landmarks, presence = self.convert_mediapipe(self.decode_image(frame))
        left_present, right_present = presence

        # If both hands are gone for >8 frames (~400ms) mid-window, drop the
        # partial buffer so it doesn't poison the embedding.
        if mode == 'translate':
            if not left_present and not right_present:
                self.both_missing_count += 1
                if self.both_missing_count > 8 and len(self.hand_landmarks) > 0:
                    self.hand_landmarks = []
                    self.presence_mask = []
                    self.both_missing_count = 0
                    return {"hands": [left_present, right_present]}
            else:
                self.both_missing_count = 0

        self.hand_landmarks.append(current_landmarks)
        self.presence_mask.append([left_present, right_present])
        if mode == 'translate':
            if len(self.hand_landmarks) == 30:
                output = self.getEmbedding(self.hand_landmarks)
                query_presence = np.array(self.presence_mask, dtype=np.uint8)
                _, costs = classify(output, self.threshold, self.database,
                                    query_presence=query_presence)
                self.hand_landmarks = []
                self.presence_mask = []

                # Debounced decode (bench-validated): a class is emitted only
                # after winning DEBOUNCE consecutive 10-frame chunks; the last
                # cost row is the no-match pseudo-class.
                names = [entry[0] for entry in self.database] + [None]
                emitted = []
                for col in np.argmin(costs, axis=0):
                    label = names[col]
                    if label == self.run_label:
                        self.run_len += 1
                    else:
                        self.run_label, self.run_len = label, 1
                    if self.run_label is not None and self.run_len == self.DEBOUNCE:
                        if self.run_label != self.prev_emitted:
                            emitted.append(self.run_label)
                            self.prev_emitted = self.run_label
                    if self.run_label is None and self.run_len >= self.DEBOUNCE:
                        self.prev_emitted = None

                if emitted:
                    self.curr_sentence.extend(emitted)
                    self.refreshCount = 0
                    if len(self.curr_sentence) > 15:
                        self.curr_sentence = self.curr_sentence[-15:]
                else:
                    self.refreshCount += 1
                    if self.refreshCount == 8:
                        self.curr_sentence.clear()
                        self.refreshCount = 0

                return {"hands": [left_present, right_present],
                        "sentence": ' '.join(self.curr_sentence)}
            return {"hands": [left_present, right_present]}

        if mode == 'record':
            if self.mouth_open(frame):
                return True

        return False

    def reset_data(self):
        self.curr_sentence.clear()
        self.prev_left_hand = np.zeros((21, 3))
        self.prev_right_hand = np.zeros((21, 3))
        self.hand_landmarks = []
        self.presence_mask = []
        self.left_missing_count = 0
        self.right_missing_count = 0
        self.both_missing_count = 0
        self.lastWord = None
        self.refreshCount = 0
        self.run_label = None
        self.run_len = 0
        self.prev_emitted = None
        # print("Reseted")

    def record(self, frame):
        return 'MOUTH_OPEN_TRUE' if self.recieve(frame, mode='record') else None

    async def stop_recording(self, name, video_data=None):
        if len(self.presence_mask) == len(self.hand_landmarks) and len(self.presence_mask) > 0:
            presence = np.array(self.presence_mask, dtype=np.uint8)
        else:
            presence = np.ones((len(self.hand_landmarks), 2), dtype=np.uint8)
        # trim blank lead-in/out so the prototype holds only the actual sign
        landmarks, presence = trim_active_span(self.hand_landmarks, presence)
        if len(landmarks) < 2:
            landmarks = np.array(self.hand_landmarks)
        embeddings = self.getEmbedding(landmarks)
        folder_path = f'server/database/{self.id}/{name}'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            with open(os.path.join(folder_path, 'description.txt'), 'w') as file:
                file.write("")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_npy = f'{timestamp}.npy'
        filename_video = f'{timestamp}.webm'
        filename_presence = f'{timestamp}.presence.npy'
        file_path_npy = os.path.join(folder_path, filename_npy)
        file_path_video = os.path.join(folder_path, filename_video)
        file_path_presence = os.path.join(folder_path, filename_presence)

        try:
            np.save(file_path_npy, embeddings)
            np.save(file_path_presence, presence)
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
        self.presence_mask = []

        self.database.append((name, embeddings, file_path_npy, file_path_video, presence))
        self.threshold = auto_threshold(self.database)
        print(f"[calibration] threshold recalibrated = {self.threshold:.3f} "
              f"({len(self.database)} prototypes)")

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
            for class_name, _, _, video_path, *_ in self.database:
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
                filename_presence = f'{filename}.presence.npy'
                file_path_npy = os.path.join(directory_path, filename_npy)
                file_path_video = os.path.join(directory_path, filename_video)
                file_path_presence = os.path.join(directory_path, filename_presence)

                if os.path.exists(file_path_npy):
                    os.remove(file_path_npy)
                    # print(f"Deleted {file_path_npy}")
                if os.path.exists(file_path_video):
                    os.remove(file_path_video)
                    # print(f"Deleted {file_path_video}")
                if os.path.exists(file_path_presence):
                    os.remove(file_path_presence)


            newdatabase = []
            for entry in self.database:
                video_path = entry[3]
                if not os.path.basename(video_path)[:-5] in toRemove:
                    newdatabase.append(entry)
            self.database = newdatabase
            self.threshold = auto_threshold(self.database)
        except Exception as e:
                return f"Error deleting files: {e}"

    def delete_folders(self, folders):
        newdatabase = []
        hash = set(folders)
        for entry in self.database:
            if not entry[0] in hash:
                newdatabase.append(entry)
        
        for class_name in folders:
            self.classDescriptions.pop(class_name)
        self.database = newdatabase
        self.threshold = auto_threshold(self.database)

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
