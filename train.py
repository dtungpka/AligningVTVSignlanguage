import torch
import torch.nn as nn
import SLDLoader.torch
import numpy as np
import random
import os
import argparse
import torch.optim as optim
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
from datetime import datetime
import csv
from torch.utils.data import DataLoader
from model import ModifiedLightweight3DCNN

def init_seed(seed):
    return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(data_folder, highlight_sign, save_path, n_frames, epochs):
    # Append '_cropped.npy' to the sign id to get the filename
    sign_filename = f"{highlight_sign}_cropped.npy"
    sign_file_path = os.path.join(data_folder, sign_filename)
    if not os.path.exists(sign_file_path):
        print(f"Data file '{sign_filename}' for sign '{highlight_sign}' not found in '{data_folder}'. Skipping...")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_path, exist_ok=True)

    # Check if model already exists
    exist_ = False
    for file in os.listdir(save_path):
        if highlight_sign in file:
            exist_ = True
            break
    if exist_:
        print(f'Model for sign {highlight_sign} already trained')
        return None

    # Prepare dataset and data loader
    dataset = SLDLoader.torch.SLD(data_folder, n_frames, 32)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset.get_generator(sign_filename, num_data=128),
        batch_size=32,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=init_seed
    )

    # Initialize model, criterion, optimizer
    model = ModifiedLightweight3DCNN(n_frames).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in tqdm(range(epochs), desc=f'Training {highlight_sign}'):
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, label)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    total, correct, TP, FP, FN = 0, 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            predicted = outputs > 0.5
            total += label.size(0)
            correct += (predicted == label).sum().item()
            TP += ((predicted == label) & (predicted == 1)).sum().item()
            FP += ((predicted != label) & (predicted == 1)).sum().item()
            FN += ((predicted != label) & (predicted == 0)).sum().item()
    accuracy = correct / total
    precision = TP / ((TP + FP) if TP + FP != 0 else 1)
    recall = TP / ((TP + FN) if TP + FN != 0 else 1)
    f1 = 2 * precision * recall / ((precision + recall) if precision + recall != 0 else 1)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    # Save model
    save_name = os.path.join(save_path, f'{highlight_sign}_{round(accuracy * 100)}_{round(precision * 100)}.pth')
    torch.save(model.state_dict(), save_name)

    # Return performance metrics
    return {'sign': highlight_sign, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def process_signs(data_folder, save_path, signs_list, epochs=50, n_frames=5):
    print("Training signs from the provided list.")
    results = []
    for sign in tqdm(signs_list, desc='Training signs'):
        metrics = train(data_folder, sign, save_path, n_frames, epochs)
        if metrics is not None:
            results.append(metrics)
    # Save performance report
    now = datetime.now()
    csv_file = f"training_results_{now.second}_{now.minute}_{now.hour}_{now.day}_{now.month}_{now.year}.csv"
    csv_path = os.path.join('results', csv_file)
    os.makedirs('results', exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['sign', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    print(f"Performance report saved to {csv_path}")

if __name__ == '__main__':
    
    # 'signs_file' is fixed
    signs_file = 'pending_training.txt'
    if not os.path.exists(signs_file):
        print(f"File '{signs_file}' does not exist.")
        exit(1)
    with open(signs_file, 'r') as f:
        signs_list = [line.strip() for line in f.readlines()]
    process_signs(data_folder="skeletons", save_path="models", signs_list=signs_list)
