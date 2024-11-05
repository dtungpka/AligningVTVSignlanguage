import torch
import torch.nn as nn
import SLDLoader.torch
import numpy as np
import random
import os
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
import csv
from torch.utils.data import DataLoader
import pickle
from model import ModifiedLightweight3DCNN

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



def compute_iou(pred_start, pred_end, true_start, true_end):
    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return intersection / union if union != 0 else 0

def compute_map(confidences, ground_truth):
    # Sort by descending confidence
    sorted_indices = np.argsort(-confidences)
    sorted_gt = ground_truth[sorted_indices]
    true_positives = np.cumsum(sorted_gt)
    false_positives = np.cumsum(1 - sorted_gt)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / np.sum(ground_truth)
    # Compute Average Precision (AP)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        if p.size > 0:
            ap += p.max()
    ap /= 11
    return ap
def compute_best_f1(confidences, ground_truth):
    thresholds = np.linspace(0, 1, num=101)
    best_f1 = 0
    best_threshold = 0
    for threshold in thresholds:
        preds = (confidences >= threshold).astype(int)
        tp = np.sum((preds == 1) & (ground_truth == 1))
        fp = np.sum((preds == 1) & (ground_truth == 0))
        fn = np.sum((preds == 0) & (ground_truth == 1))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold

def test(Xs, check_point, frame_count=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModifiedLightweight3DCNN(in_channels=frame_count)
    model.load_state_dict(torch.load(check_point))
    model.eval().to(device)
    with torch.no_grad():
        data = torch.tensor(Xs, dtype=torch.float32).to(device)
        data = torch.einsum('b t w c -> b t c w', data)
        output = model(data)
        result_list = output.cpu().numpy()
    return result_list

def process_word(task, model_paths, video, save_path, frame_count=5, delta=120):
    start_gt = task['start_frame']
    end_gt = task['end_frame']
    start_frame = start_gt - delta
    end_frame = end_gt + delta
    #print(f"Processing {task['word']} from {start_frame} to {end_frame}")
    #print(f"Model ids: {task['ids']}")

    valid_model_paths = []
    for _m in task['ids']:
        for m in model_paths:
            if _m in m:
                info = m.replace('.pth','').split('_')
                if len(info) < 2:
                    continue
                acc, perc = info[-2], info[-1]
                if int(perc) < 40 and int(acc) < 40:
                    print(f"Model {m} is not good enough, skipping..")
                    continue
                _p = os.path.join(checkpoint_path, m)
                if os.path.exists(_p):
                    valid_model_paths.append(_p)
                break
    if not valid_model_paths:
        #print('No valid model found, skipping..')
        return None

    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, len(video))
    sequence = video[start_frame:end_frame]

    Xs = [np.array([conv_skeleton(frame, mode='CORDTOLOCAL') for frame in sequence[i:i+frame_count]])
          for i in range(len(sequence) - frame_count + 1)]
    Xs = np.array(Xs)

    # Build ground truth labels per sample
    ground_truth = []
    for i in range(len(sequence) - frame_count + 1):
        sample_start = start_frame + i
        sample_end = sample_start + frame_count - 1
        overlap = max(0, min(sample_end, end_gt) - max(sample_start, start_gt) + 1)
        ground_truth.append(1 if overlap > 0 else 0)
    ground_truth = np.array(ground_truth)

    results = []
    for check_point in valid_model_paths:
        output = test(Xs, check_point, frame_count)
        confidence = np.max(output, axis=1)
        best_f1, best_threshold = compute_best_f1(confidence, ground_truth)

        # Use the optimal threshold to determine predictions
        pred_indices = np.where(confidence >= best_threshold)[0]
        if pred_indices.size > 0:
            pred_start = start_frame + pred_indices[0]
            pred_end = start_frame + pred_indices[-1] + frame_count
        else:
            pred_start, pred_end = start_frame, start_frame

        iou = compute_iou(pred_start, pred_end, start_gt, end_gt)
        map_score = compute_map(confidence, ground_truth)
        results.append({'model': os.path.basename(check_point),
                        'iou': iou,
                        'mAP': map_score,
                        'f1': best_f1,
                        'threshold': best_threshold})

    return results

def process_evaluation_pack(evaluation_pack, checkpoint_path, save_path, frame_count=5, delta=120, csv_path='results.csv'):
    model_paths = os.listdir(checkpoint_path)
    skeletons = {name: data for name, data in evaluation_pack['skeletons']}
    records = []

    for word, details in tqdm(evaluation_pack['words'].items(), desc='Processing words'):
        for occ in details['occurences']:
            video_name, start_frame, end_frame, ids = occ
            video = skeletons.get(video_name)
            if video is None:
                #print(f"Video {video_name} not found, skipping..")
                continue
            task = {
                'word': word,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'ids': ids
            }
            result = process_word(task, model_paths, video, save_path, frame_count, delta)
            if result:
                for res in result:
                    records.append({
                        'word': word,
                        'video': video_name,
                        'model': res['model'],
                        'iou': res['iou'],
                        'mAP': res['mAP'],
                        'f1': res['f1'],
                        'threshold': res['threshold']
                    })

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, csv_path), 'w', newline='') as csvfile:
        fieldnames = ['word', 'video', 'model', 'iou', 'mAP', 'f1', 'threshold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
if __name__ == '__main__':
    # Set variables for notebook use
    evaluation_pack_path = 'evaluation_pack.pkl'
    checkpoint_path = 'models'
    save_path = 'results'
    frame_count = 5
    delta = 120

    # Load evaluation pack and process
    with open(evaluation_pack_path, 'rb') as f:
        evaluation_pack = pickle.load(f)

    process_evaluation_pack(evaluation_pack, checkpoint_path, save_path, frame_count, delta)