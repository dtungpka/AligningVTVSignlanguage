import mediapipe as mp
import cv2
import os
import logging
import time
import json
import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle as pkl
import numpy as np
import datetime
import sys


# Create a folder for logs
os.makedirs("logs", exist_ok=True)

import logging.handlers
# Set up logging
log_filename = datetime.datetime.now().strftime("dn_%d_%m_%H_%M_%S.log")
log_filepath = os.path.join("logs", log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(log_filepath, maxBytes=(1048576*50), backupCount=7),
        logging.StreamHandler(sys.stdout)
    ],force=True
)

def handle_exception(exc_type, exc_value, exc_traceback):
    # Custom exception handling logic here
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

class main:
    def __init__(self,line_thickness=2,circle_radius=2) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_draw_spec = self.mp_drawing.DrawingSpec(thickness=line_thickness, circle_radius=circle_radius, color=(255,0,0))

        #MP hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.hand_draw_spec = self.mp_drawing.DrawingSpec(thickness=line_thickness, circle_radius=circle_radius, color=(0,255,0))

        #MP face
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh()
        '''
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
                                            '''
        
    def get_skeleton(self,video:str, output_path:str,npy_path:str):
        #Get video from path, apply mp_pose, mp_hands, mp_face and draw skeleton using cv2, save to output_path. Skeleton data is saved to npy_path as a json file
        #Each mmp have a different color: pose: red, hands: green, face: blue
        #Get video from path
        assert os.path.isfile(video), f'Video {video} not found'
        cap = cv2.VideoCapture(video)
        #Get video name
        video_name = video.replace('/','\\').split('\\')[-1].split('.')[0]
        #Get video fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        #Get video size
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #Get video length
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Check if output video already exists and frame count is equal to input video, if yes, skip
        if os.path.isfile(os.path.join(output_path,f'{video_name}.avi')) and os.path.isfile(os.path.join(npy_path,f'{video_name}.npy')):
            cap2 = cv2.VideoCapture(os.path.join(output_path,f'{video_name}.avi'))
            frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count2 == frame_count:
                print(f'Video {video_name} already exists, skipping...')
                cap.release()
                cap2.release()
                return
            else:
                cap2.release()
                #Delete old video
                os.remove(os.path.join(output_path,f'{video_name}.avi'))
                os.remove(os.path.join(npy_path,f'{video_name}.json'))
                print(f'Video {video_name} already exists but frame count is different, overwriting...')

        last_pose_landmarks = np.zeros((33, 2))
        last_hand_landmarks = np.zeros((21*2, 2))           
        pose_landmarks_list = []
        hand_landmarks_list = []                                                        
        #Get video duration
        duration = frame_count/fps
        #Get video codec
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        #Create video writer
        out = cv2.VideoWriter(os.path.join(output_path,f'{video_name}.avi'), codec, fps, (width, height))
        #Create progress bar
        logging.info(f'Processing video {video_name} with {frame_count} frames')
        start_t = time.time()
        #Start processing, loop through each frame and apply mp_pose, mp_hands, mp_face
        for i in range(frame_count):
            #Read frame
            success, image = cap.read()
            #Check if frame is valid
            if not success:
                logging.error(f'Error reading frame {i} from video {video}')
                continue
            #Convert image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Process pose
            pose_results = self.pose.process(image)
            #Process hands
            hands_results = self.hands.process(image)
            #Process face
            #face_results = self.face.process(image)
            #Draw pose
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.pose_draw_spec, self.pose_draw_spec)
            #Draw hands
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.hand_draw_spec, self.hand_draw_spec)
            #Draw face
            #if face_results.multi_face_landmarks:
            #    for face_landmarks in face_results.multi_face_landmarks:
            #        self.mp_drawing.draw_landmarks(image, face_landmarks, self.mp_face.FACE_CONNECTIONS, self.face_draw_spec, self.face_draw_spec)
            #Write image to video
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            #Update progress bar
            #Save data to json

            #Convert landmarks to dict
            pose_landmarks = last_pose_landmarks
            if pose_results.pose_landmarks:
                pose_landmarks = np.array([(lm.x, lm.y) for lm in pose_results.pose_landmarks.landmark])
                last_pose_landmarks = pose_landmarks
            pose_landmarks_list.append(pose_landmarks)
            if hands_results.multi_hand_landmarks:
                hand_landmarks = last_hand_landmarks.copy()
                for idx, handLms in enumerate(hands_results.multi_hand_landmarks):
                    hand_idx = 0 if hands_results.multi_handedness[idx].classification[0].label == 'Left' else 1
                    hand_landmarks[hand_idx*21:(hand_idx+1)*21] = np.array([(lm.x, lm.y) for lm in handLms.landmark])


                #hand_landmarks = np.array([[(lm.x, lm.y) for lm in hand.landmark] for hand in hand_results.multi_hand_landmarks])
                hand_landmarks = hand_landmarks.reshape(-1, 2)


                last_hand_landmarks = hand_landmarks
            else:
                hand_landmarks = last_hand_landmarks
            hand_landmarks_list.append(hand_landmarks)
        pose_landmarks_array = np.array(pose_landmarks_list)
        hand_landmarks_array = np.array(hand_landmarks_list)

        #fill all 0 values with nearest non-zero value
        last_non_zero_pose = np.zeros((33,2))
        for t in range(pose_landmarks_array.shape[0]-1,-1,-1):
            if np.all(pose_landmarks_array[t]==0):
                pose_landmarks_array[t] = last_non_zero_pose
            else:
                last_non_zero_pose = pose_landmarks_array[t]

        last_non_zero_hand_left = np.zeros((21,2))
        last_non_zero_hand_right = np.zeros((21,2))

        for t in range(hand_landmarks_array.shape[0]-1,-1,-1):
            #check each hand
            if np.all(hand_landmarks_array[t,:21]==0):
                hand_landmarks_array[t,:21] = last_non_zero_hand_left
            else:
                last_non_zero_hand_left = hand_landmarks_array[t,:21]
            if np.all(hand_landmarks_array[t,21:]==0):
                hand_landmarks_array[t,21:] = last_non_zero_hand_right
            else:
                last_non_zero_hand_right = hand_landmarks_array[t,21:]





        logging.info(f'Processing video {video_name} took {time.time()-start_t:.2f}s')
        os.makedirs(npy_path, exist_ok=True)
        np.save(os.path.join(npy_path,video_name+'.npy'), np.concatenate((pose_landmarks_array, hand_landmarks_array), axis=1))
        #Release video
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f'Video {video_name} processed successfully!')




    def process_videos(self,input_video_folder,output_video_folder):
        out_videos_path = os.path.join(output_video_folder,'videos')
        skeletons_path = os.path.join(output_video_folder,'skeletons')
        os.makedirs(out_videos_path, exist_ok=True)
        os.makedirs(skeletons_path, exist_ok=True)
        test_only_videos = []#[f'src{i}' for i in range(1,10)]
        #Transverse all video in input_video_folder
        full_path_videos = {}
        for root, dirs, files in os.walk(input_video_folder):
            for file in files:
                if file.endswith('.mp4') or file.endswith('.avi'):
                    skip = test_only_videos != [] #If test_only_videos is empty, skip is False, else skip is True
                    for test_only_video in test_only_videos:
                        if test_only_video in file:
                            skip = False
                            print(f'Adding {file}')
                            break
                    if skip:
                        continue
                    full_path_videos[file] = os.path.join(root, file)
        print(f'Found {len(full_path_videos)} videos')
        #Process each video
        for video_name, video_path in full_path_videos.items():
            self.get_skeleton(video_path,out_videos_path,skeletons_path)
        print('Done!')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Process videos and extract skeletons.')
    parser.add_argument('input_video_folder', type=str,default="SLD\cropped", help='Path to the folder containing input videos')
    parser.add_argument('output_video_folder', type=str,default="SLD\output", help='Path to the folder to save output videos')
    args = parser.parse_args()

    main().process_videos(args.input_video_folder, args.output_video_folder)
    





