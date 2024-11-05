import tqdm
import random
import os
import cv2
import numpy as np
import json
import math
import torch
from sklearn.cluster import KMeans
LAST_LOADED_NPY = {}


def filter_mask(data):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit_predict(data.reshape(-1,1))
    return kmeans
    

def average_data(data):
    return np.mean(data, axis=0)
def compare_frame(frame1,frame2):
    delta = np.abs(frame1 - frame2)
    score = np.sqrt(delta[:,0]**2 + delta[:,1]**2,)
    return np.sum(score)

def get_action_time_mask(data):
    #2 mask, front and back using kmeans
    no_of_samples = max(data.shape[0]//10,10)
    sample_idle_frame_front = average_data(data[5:no_of_samples,:,:])
    sample_idle_frame_back = average_data(data[-no_of_samples:-5,:,:])
    
    scores_front = []
    scores_back = []
    for i in range(data.shape[0]):
        r_front = compare_frame(data[i],sample_idle_frame_front)
        r_back = compare_frame(data[i],sample_idle_frame_back)
        #print(f"Score {i}: {r_front} {r_back}")
        scores_front.append(r_front)
        scores_back.append(r_back)
    scores_front = np.array(scores_front)
    scores_back = np.array(scores_back)
    mask_front = filter_mask(scores_front)
    mask_back = filter_mask(scores_back)
    #inverse the mask if the first frame is not idle
    if mask_front[0] == 1:
         mask_front = 1 - mask_front
         mask_back = 1 - mask_back
    mask = mask_front * mask_back
    #print(f"Threshold: {threshold_front} {threshold_back}")
    #mask = np.array([1 if score_front > threshold_front and score_back > threshold_back else 0 for score_front,score_back in zip(scores_front,scores_back)])
    return mask
def crop_from_mask(data, mask):
    start = 0
    end = 0
    for i in range(mask.shape[0]):
        if mask[i] == 1:
            start = i
            break
    for i in range(mask.shape[0]-1,0,-1):
        if mask[i] == 1:
            end = i
            break
    return data[start:end]

#crop -> augme -> cov

def conv_skeleton(skeleton,mode='CORDTOLOCAL',upper_body_only=True):
    if upper_body_only:
        #to 42:42+23 if upper_body_only else keep all
        skeleton = skeleton[:42+23,:]
    anchor_point = [0,21,42,skeleton.shape[0]]
    result_skeleton = np.zeros_like(skeleton)
    if mode == 'CORDTOLOCAL':
        #convert global cordiante of 1:21 to local with anchor point 0
        for i,anchor in enumerate(anchor_point[:-1]):
            result_skeleton[anchor_point[i]:anchor_point[i+1]] = skeleton[anchor_point[i]:anchor_point[i+1]] - skeleton[anchor]
            #normalize the data to -1,1
            local_skeleton = result_skeleton[anchor_point[i]:anchor_point[i+1]]

            result_skeleton[anchor_point[i]:anchor_point[i+1],0] = local_skeleton[:,0] / np.max(np.abs(local_skeleton[:,0]))
            result_skeleton[anchor_point[i]:anchor_point[i+1],1] = local_skeleton[:,1] / np.max(np.abs(local_skeleton[:,1]))
            #store the original anchor point cord
            result_skeleton[anchor] = skeleton[anchor]
    
    return result_skeleton

class SLD:
    def __init__(self,dataset_path,n_frame=30,batch_size=1,random_seed=42) -> None:
        '''
        Sign Language Dataset Loader (Preprocessing and Data Augmentation)
        '''
        self.dataset_path = dataset_path
       
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_frames = n_frame



        
    def get_generator(self,highlight_word="",num_data=100):
        return Generator(self.dataset_path,highlight_word,self.batch_size,self.random_seed,n_frames=self.n_frames,num_data=num_data)
        
class Generator(torch.utils.data.IterableDataset):
    def __init__(self,data_paths,highlight_word,batch_size,random_seed,n_frames,num_data) -> None:
        
        self.data_paths = data_paths
        self.highlight_word = highlight_word
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.random_seed = random_seed
        self.num_data = num_data
        self.full_data_list = os.listdir(data_paths)
        #remove highlight word from full_data_list
        self.full_data_list.remove(highlight_word)
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)
    def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(self.get_data(iter_start, iter_end))
    def augment_data(self,data,frame_skip,time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift,out_frames = 30):
                # Existing augmentations...

        # Random limb length scaling
        limb_scaling_factors = np.random.uniform(0.9, 1.1, size=(1, data.shape[1], data.shape[2]))
        data = data * limb_scaling_factors

        # Random joint angle perturbations
        joint_angle_perturbations = np.random.uniform(-0.05, 0.05, size=data.shape)
        data = data + joint_angle_perturbations

        # Simulate missing hand or partial skeleton
        if np.random.rand() < 0.5:
            # Remove one hand (either left or right)
            hand_to_remove = np.random.choice(['left', 'right'])
            if hand_to_remove == 'left':
                data[:, :21, :] = 0  # Assuming first 21 keypoints are left hand
            else:
                data[:, 21:42, :] = 0  # Assuming next 21 keypoints are right hand

        # Continue with existing augmentations...
        #Time crop is 1-3, first we div the video to 7 parts, then we crop part:
        #1: 1 - 3
        #2: 2 - 4
        #3: 3 - 5

        #if frame skip < 0, add frame by one (np.repeat)
        #if frame skip > 0, remove frame so duration /2
        _data = []
        frame_skip += 1
        while len(_data) < 7:
            frame_skip -= 1
            #print(f"Frame Skip: {frame_skip}")
            if frame_skip < 0:
                _data = np.repeat(data, abs(frame_skip), axis=0)
            elif frame_skip > 0:
                _data = data[frame_shift::frame_skip]
            elif frame_skip < -5 or frame_skip == 0:
                _data = data
                break
        data = _data
        #Get the duration of the video
        duration = data.shape[0]
        crop_parts = 7
        time_crop_variant = time_crop // 6
        time_crop = time_crop % 6
        if time_crop >= 0 and time_crop_variant >= 0:
            while time_crop >= 0:
                crop_duration = duration // crop_parts
                if crop_duration > 0:
                    start =  crop_duration * time_crop
                    end = min(crop_duration * (time_crop + time_crop_variant + 2), duration)
                    #print(f"Start: {start} End: {end} Duration: {duration}, time_crop: {time_crop}, time_crop_variant: {time_crop_variant}")
                    _data = data[start:end]
                    break
                else:
                    time_crop -= 1
            else:
                _data = data
        data = _data
        #center data to 0-1
        # try:
        #     data = data - np.min(data)
        # except: 
        #     print(f" {data.shape}:")
        #     print(f"{frame_skip}: {_data.shape}")

        # #makesure the data is in 0-1 in x and y axis
        # data[:, :, 0] = data[:, :, 0] / np.max(data[:, :, 0])
        # data[:, :, 1] = data[:, :, 1] / np.max(data[:, :, 1])

        #print(f"Min: {np.min(data)} Max: {np.max(data)}, Shape: {data.shape}")

        # Zoom
        data_zoomed = data * zoom_factor

        # Rotate
        center = (np.max(data_zoomed, axis=(0, 1)) - np.min(data_zoomed, axis=(0, 1))) / 2
        data_centered = data_zoomed - center
        # Shift (move)
        data_rotated = np.dot(data_centered, rotation_matrix.T)
        data_shifted = data_rotated + center

        #shift every point
        for i in range(data_rotated.shape[1]):
            shift_value = np.random.uniform( shift_values - shift_values/5, shift_values + shift_values/5, 2)
            data_shifted[:,i] = data_shifted[:,i] + shift_value

        #print(f"Outframes: {out_frames} Data: {data_shifted.shape}")
        #Np.repeat and slice to out_frames
        if out_frames > data_shifted.shape[0]:
            data_shifted = np.repeat(data_shifted, out_frames // data_shifted.shape[0] + 1, axis=0)
        data_shifted = data_shifted[:out_frames]
        #print(np.max(data_shifted), np.min(data_shifted))
        return data_shifted
    def get_augmented_data(self,data):
        rotation_angle = np.random.uniform(-.1, .1)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle),  np.cos(rotation_angle)]])
        zoom_factor = np.random.uniform(0.8, 1.2)
        shift_values = np.random.uniform(-0.05, 0.05, 1)
        speed = np.random.choice([ 0, 1, 2, 3, 4])
        time_crop = np.random.choice([-1,-1, 0, 1, 2, 3])
        frame_shift = np.random.randint(0, 18)
        return self.augment_data(data, speed, time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift=frame_shift, out_frames=self.n_frames)

    def get_data(self,start,end):
        global LAST_LOADED_NPY
        #i = 0
        last_true_label = 9999
        for i in range(self.num_data):

            should_load_true_label = np.random.choice([True, False], p=[0.3, 0.7])
            # if not should_load_true_label and last_true_label > self.num_data // 10:
            #     last_true_label = 0
            #     should_load_true_label = True

            if should_load_true_label:
                data_point_path = os.path.join(self.data_paths,  self.highlight_word)
                if data_point_path in LAST_LOADED_NPY:
                    cropped_data = LAST_LOADED_NPY[data_point_path]
                else:
                    data = np.load(data_point_path)
                    mask = get_action_time_mask(data)
                    cropped_data = crop_from_mask(data, mask)
                    #if shape 0 cropped data is = 0, print mask
                    if cropped_data.shape[0] == 0:
                        #print(f'{data_point_path} cannot cluster, defaulting to full data')
                        cropped_data = data
                    LAST_LOADED_NPY[data_point_path] = cropped_data
                last_true_label = 0
                y = 1
            else:
                data_point_path = os.path.join(self.data_paths, random.choice(self.full_data_list))
                if data_point_path in LAST_LOADED_NPY:
                    cropped_data = LAST_LOADED_NPY[data_point_path]
                else:
                    data = np.load(data_point_path)
                    mask = get_action_time_mask(data)
                    cropped_data = crop_from_mask(data, mask)
                    #if shape 0 cropped data is = 0, print mask
                    if cropped_data.shape[0] == 0:
                        #print(f'{data_point_path} cannot cluster, defaulting to full data')
                        cropped_data = data
                    else:
                        pass
                        #print(f'{data_point_path} cluster success')
                    LAST_LOADED_NPY[data_point_path] = cropped_data
                last_true_label += 1
                y = 0

            
            augmented_data = self.get_augmented_data(cropped_data)
            r = np.array([conv_skeleton(augmented_data[i],mode='CORDTOLOCAL') for i in range(augmented_data.shape[0])])

            # print(f"Data: {r.shape} {y}")
            X = torch.FloatTensor(r)
            y = torch.FloatTensor([y])
            yield X, y
    def __iter__(self):
        return self()
    def __len__(self):
        return self.num_data
    def __call__(self):
        return self.get_data(0,self.num_data)