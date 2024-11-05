
import cv2
import os
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import multiprocessing
from multiprocessing import Pool
import tqdm
import time
import numpy as np

def augment_data(data,frame_skip,time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift,out_frames = 30):

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
        if frame_skip < 0:
            _data = np.repeat(data, abs(frame_skip), axis=0)
        elif frame_skip > 0:
            _data = data[frame_shift::frame_skip]
            
    data = _data
    #Get the duration of the video
    duration = data.shape[0]
    crop_duration = duration // 7
    start =  crop_duration * time_crop
    end = start + crop_duration * 2
    data = data[start:end]
    

    #center data to 0-1
    try:
        data = data - np.min(data)
    except: 
        print(f" {data.shape}:")
        print(f"{frame_skip}: {_data.shape}")

    #makesure the data is in 0-1 in x and y axis
    data[:, :, 0] = data[:, :, 0] / np.max(data[:, :, 0])
    data[:, :, 1] = data[:, :, 1] / np.max(data[:, :, 1])

    print(np.max(data), np.min(data))

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


    #Np.repeat and slice to out_frames
    if out_frames > data_shifted.shape[0]:
        data_shifted = np.repeat(data_shifted, out_frames // data_shifted.shape[0] + 1, axis=0)
    data_shifted = data_shifted[:out_frames]
    print(np.max(data_shifted), np.min(data_shifted))
    return data_shifted


#----- EDIT HERE -----
def get_augmented_data(data):
    rotation_angle = np.random.uniform(-.1, .1)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                        [np.sin(rotation_angle),  np.cos(rotation_angle)]])
    zoom_factor = np.random.uniform(0.8, 1.2)
    shift_values = np.random.uniform(-0.1, 0.1, 1)
    speed = np.random.choice([ 0, 1, 2, 3, 4])
    time_crop = np.random.choice([1, 2, 3])
    frame_shift = np.random.choice([ 0,0, 1, 2, 3, 4,5])
    return augment_data(data, speed, time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift=frame_shift, out_frames=30)
#--------------------

def visualize(data,variation):
    #os.makedirs('visualize', exist_ok=True)
    for t in range(data.shape[0]):
        image = np.zeros((512,512,3),dtype=np.uint8)+255
        #for each point, draw a circle
        for point in range(data.shape[1]):
            x = int(data[t,point,0]*512)
            y = int(data[t,point,1]*512)
            image = cv2.circle(image, (x,y), 2, (255,0,0), -1)
        #cv2.imwrite(f'visualize/{t}.png', image)
        cv2.putText(image, f"Variation: {variation}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', image)
        
        cv2.waitKey(int(1/25*1000))
    

data = np.load(r"sample.npy")
data.shape
variation = 0
while variation < 100:
    t_st = time.time()
    augmented_data = get_augmented_data(data)
    print('Time:', time.time()-t_st)
    visualize(augmented_data,variation)
    variation += 1
    
cv2.destroyAllWindows()