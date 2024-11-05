import tqdm
import random
import os
import cv2
import numpy as np
import json
import tensorflow as tf






class SLD:
    def __init__(self,dataset_path,n_frame=30,batch_size=1,random_seed=42) -> None:
        '''
        Sign Language Dataset Loader (Preprocessing and Data Augmentation)
        '''
        self.dataset_path = dataset_path
       
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_frames = n_frame
        self.last_loaded_npy = {}



        
    def get_generator(self,highlight_word="",num_data=100):
        return self.Generator(self.dataset_path,highlight_word,self.batch_size,self.random_seed,n_frames=self.n_frames,num_data=num_data,last_loaded_npy=self.last_loaded_npy)
        
    class Generator:
        def __init__(self,data_paths,highlight_word,batch_size,random_seed,n_frames,num_data,last_loaded_npy) -> None:
            
            self.data_paths = data_paths
            self.highlight_word = highlight_word
            self.batch_size = batch_size
            self.n_frames = n_frames
            self.random_seed = random_seed
            self.num_data = num_data
            self.full_data_list = os.listdir(data_paths)
            #remove highlight word from full_data_list
            self.full_data_list.remove(f"{highlight_word}_cropped.npy")
            self.last_loaded_npy = last_loaded_npy
            tf.random.set_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

        def augment_data(self,data,frame_skip,time_crop, zoom_factor, rotation_matrix, shift_values,out_frames = 30):

            #Time crop is 1-3, first we div the video to 7 parts, then we crop part:
            #1: 1 - 3
            #2: 2 - 4
            #3: 3 - 5

            #Get the duration of the video
            duration = data.shape[0]
            crop_duration = duration // 7
            start =  crop_duration * time_crop
            end = start + crop_duration * 2
            data = data[start:end]


            #if frame skip < 0, add frame by one (np.repeat)
            #if frame skip > 0, remove frame so duration /2
            if frame_skip < 0:
                data = np.repeat(data, abs(frame_skip), axis=0)
            elif frame_skip > 0:
                data = data[::frame_skip]

            

            #center data to 0-1
            data = data - np.min(data)
            data = data / np.max(data)
            

            # Zoom
            data_zoomed = data * zoom_factor

            # Rotate
            center = (np.max(data_zoomed, axis=(0, 1)) - np.min(data_zoomed, axis=(0, 1))) / 2
            data_centered = data_zoomed - center
            # Shift (move)
            data_rotated = np.dot(data_centered, rotation_matrix.T)
            data_shifted = data_rotated + center + shift_values

            #Np.repeat and slice to out_frames
            if out_frames > data_shifted.shape[0]:
                data_shifted = np.repeat(data_shifted, out_frames // data_shifted.shape[0] + 1, axis=0)
            data_shifted = data_shifted[:out_frames]

            return data_shifted
        def get_augmented_data(self,data):
            rotation_angle = np.random.uniform(-10, 10)
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle),  np.cos(rotation_angle)]])
            zoom_factor = np.random.uniform(0.8, 1.2)
            shift_values = np.random.uniform(-0.1, 0.1, 2)
            speed = np.random.choice([-1, 0, 1, 2, 3])
            time_crop = np.random.choice([1, 2, 3])
            return self.augment_data(data, speed, time_crop, zoom_factor, rotation_matrix, shift_values, out_frames=self.n_frames)

        def __call__(self):

            #i = 0
            last_true_label = 9999
            for i in range(self.num_data):

                should_load_true_label = np.random.choice([True, False], p=[0.5, 0.5])
                if not should_load_true_label and last_true_label > self.num_data // 10:
                    last_true_label = 0
                    should_load_true_label = True

                if should_load_true_label:
                    data_point_path = os.path.join(self.data_paths,  f"{self.highlight_word}_cropped.npy")
                    if data_point_path in self.last_loaded_npy:
                        data = self.last_loaded_npy[data_point_path]
                    else:
                        data = np.load(data_point_path)
                    last_true_label = 0
                    y = 1
                else:
                    data_point_path = os.path.join(self.data_paths, random.choice(self.full_data_list))
                    if data_point_path in self.last_loaded_npy:
                        data = self.last_loaded_npy[data_point_path]
                    else:
                        data = np.load(data_point_path)
                    last_true_label += 1
                    y = 0


                augmented_data = self.get_augmented_data(data)

                #convert to tf
                X = tf.convert_to_tensor(augmented_data)
                yield X, y
        def __iter__(self):
            return self()
