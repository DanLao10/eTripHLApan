#!/usr/bin/env python3
"""
Testing script for Extended TripHLApan model
"""

import sys
import os

# Change to codes directory
script_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.join(script_dir, 'codes')
os.chdir(codes_dir)
sys.path.insert(0, codes_dir)

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, matthews_corrcoef
)
from datetime import datetime

# Import from TripHLApan codes
from helper import *
from data_pre_processing2 import *

# Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print(f"Using device: {DEVICE}")

# Set random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if USE_CUDA:
    torch.cuda.manual_seed(random_seed)

# Network definition (same as original)
class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()

        # network1
        self.gru1 = nn.GRU(20, 128, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(20, 128, batch_first=True, bidirectional=True)
        self.full_conn1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network2
        self.embedding1 = nn.Embedding(21, 6)
        self.embedding2 = nn.Embedding(21, 6)
        self.gru3 = nn.GRU(6, 128, batch_first=True, bidirectional=True)
        self.gru4 = nn.GRU(6, 128, batch_first=True, bidirectional=True)
        self.full_conn2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network3
        self.gru5 = nn.GRU(28, 128, batch_first=True, bidirectional=True)
        self.gru6 = nn.GRU(28, 128, batch_first=True, bidirectional=True)
        self.full_conn3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        self.attention1 = nn.MultiheadAttention(256, 1)
        self.attention2 = nn.MultiheadAttention(256, 1)
        self.attention3 = nn.MultiheadAttention(256, 1)
        self.attention4 = nn.MultiheadAttention(256, 1)
        self.attention5 = nn.MultiheadAttention(256, 1)
        self.attention6 = nn.MultiheadAttention(256, 1)

        # fully_conn
        self.full_conn = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, peps1, alleles1, peps2, alleles2, peps3, alleles3):
        x1 = self.gru1(peps1)[0]
        query = x1.permute(1, 0, 2)
        x_attention, __ = self.attention1(query, query, query)
        x1 = x_attention.permute(1, 0, 2)[:,-1]

        x2 = self.gru2(alleles1)[0]
        query = x2.permute(1, 0, 2)
        y_attention, __ = self.attention2(query, query, query)
        x2 = y_attention.permute(1, 0, 2)[:,-1]

        x3 = torch.cat((x1, x2), 1)
        result1 = self.full_conn1(x3)

        # network2
        X1 = self.embedding1(peps2)
        Y1 = self.embedding2(alleles2)

        x1 = self.gru3(X1)[0]
        query = x1.permute(1, 0, 2)
        x_attention, __ = self.attention3(query, query, query)
        x1 = x_attention.permute(1, 0, 2)[:,-1]

        x2 = self.gru4(Y1)[0]
        query = x2.permute(1, 0, 2)
        y_attention, __ = self.attention4(query, query, query)
        x2 = y_attention.permute(1, 0, 2)[:,-1]

        x3 = torch.cat((x1, x2), 1)
        result2 = self.full_conn2(x3)

        # network3
        x1 = self.gru5(peps3)[0]
        query = x1.permute(1, 0, 2)
        x_attention, __ = self.attention5(query, query, query)
        x1 = x_attention.permute(1, 0, 2)[:,-1]

        x2 = self.gru6(alleles3)[0]
        query = x2.permute(1, 0, 2)
        y_attention, __ = self.attention6(query, query, query)
        x2 = y_attention.permute(1, 0, 2)[:,-1]

        x3 = torch.cat((x1, x2), 1)
        result3 = self.full_conn3(x3)

        # fully_conn
        x = torch.cat((result1, result2, result3), 1)
        result = self.full_conn(x)
        return result


# Dataset class
class TripHLApanDataset(Dataset):
    def __init__(self, data_file_path):
        self.pep_seq_list = []
        self.allele_seq_list = []
        self.label_list = []
        
        self.dict_allele_seq = map_allele_name_seq()
        
        with open(data_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    pep = parts[0]
                    hla = parts[1]
                    label = int(parts[3])
                    
                    self.pep_seq_list.append(pep)
                    self.allele_seq_list.append(hla)
                    self.label_list.append(label)
        
        print(f"Loaded {len(self.pep_seq_list)} samples from {data_file_path}")
        
        self._preprocess_sequences()
    
    def _preprocess_sequences(self):
        processed_peps = []
        processed_alleles = []
        
        for pep in self.pep_seq_list:
            peplen = len(pep)
            insert_idx = int((peplen + 1) / 2) - 1
            pseq_pep_seq = pep[0:insert_idx] + 'X' * (14 - peplen) + pep[insert_idx:]
            processed_peps.append(pseq_pep_seq)
        
        for allele in self.allele_seq_list:
            if allele in self.dict_allele_seq:
                seq = self.dict_allele_seq[allele]
            else:
                seq = 'X' * 200
            seq = seq[0:200]
            seq = seq + 'X' * (200 - len(seq))
            processed_alleles.append(seq)
        
        self.pep_seq_list = processed_peps
        self.allele_seq_list = processed_alleles
    
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, idx):
        pep = self.pep_seq_list[idx]
        allele = self.allele_seq_list[idx]
        label = self.label_list[idx]
        
        # pep_encoded1 = encode_seq(pep, 'one-hot')
        # allele_encoded1 = encode_seq(allele, 'one-hot')
        pep_encoded1 = encode_seq(pep, 'BLOSUM62')
        allele_encoded1 = encode_seq(allele, 'BLOSUM62')
        pep_encoded2 = encode_seq(pep, 'num')
        allele_encoded2 = encode_seq(allele, 'num')
        pep_encoded3 = encode_seq(pep, 'AAfea_phy')
        allele_encoded3 = encode_seq(allele, 'AAfea_phy')
        
        return {
            'pep1': torch.FloatTensor(pep_encoded1),
            'allele1': torch.FloatTensor(allele_encoded1),
            'pep2': torch.LongTensor(pep_encoded2),
            'allele2': torch.LongTensor(allele_encoded2),
            'pep3': torch.FloatTensor(pep_encoded3),
            'allele3': torch.FloatTensor(allele_encoded3),
            'label': torch.FloatTensor([label])
        }


# Configuration
TEST_DATA_PATH = '../for_prediction/test_data.txt'
MODEL_DIR = '../../models/TripHLApan_extended/'
RESULTS_DIR = os.path.join(MODEL_DIR, 'test_results/')
BATCH_SIZE = 512

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("TripHLApan Extended Model Testing")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print("="*70)

# Load test dataset
print("\nLoading test dataset...")
test_dataset = TripHLApanDataset(TEST_DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load model
print(f"\nLoading model...")
model_path = os.path.join(MODEL_DIR, 'best_model_epoch2.pt')

if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    sys.exit(1)

model = Network_conn().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
print(f"Model loaded from: {model_path}")

# Test model
print("\nRunning inference on test set...")
model.eval()

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        peps1 = batch['pep1'].to(DEVICE)
        alleles1 = batch['allele1'].to(DEVICE)
        peps2 = batch['pep2'].to(DEVICE)
        alleles2 = batch['allele2'].to(DEVICE)
        peps3 = batch['pep3'].to(DEVICE)
        alleles3 = batch['allele3'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        outputs = model(peps1, alleles1, peps2, alleles2, peps3, alleles3)
        
        probabilities = outputs.cpu().numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)
        batch_labels = labels.cpu().numpy().flatten().astype(int)
        
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        all_labels.extend(batch_labels)

predictions = np.array(all_predictions)
probabilities = np.array(all_probabilities)
labels = np.array(all_labels)

# Compute metrics
print("\n" + "="*70)
print("TEST RESULTS - EXTENDED MODEL (EPOCH 74)")
print("="*70)

accuracy = accuracy_score(labels, predictions)
auc = roc_auc_score(labels, probabilities)
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)
mcc = matthews_corrcoef(labels, predictions)

tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
specificity = tn / (tn + fp)
sensitivity = recall

print(f"\nAccuracy:        {accuracy:.4f}")
print(f"AUC (ROC):       {auc:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"F1-Score:        {f1:.4f}")
print(f"MCC:             {mcc:.4f}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

print(f"\nSensitivity:     {sensitivity:.4f}")
print(f"Specificity:     {specificity:.4f}")

print(f"\nClassification Report:")
print(classification_report(labels, predictions, target_names=['Negative', 'Positive']))

# Save results
results_file = os.path.join(RESULTS_DIR, 'test_metrics_extended.txt')
with open(results_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("TripHLApan Extended Model - Test Results\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: best_model_epoch2.pt\n")
    f.write(f"Test data: {TEST_DATA_PATH}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"METRICS:\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"AUC:       {auc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"\nCONFUSION MATRIX:\n")
    f.write(f"True Negatives:  {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives:  {tp}\n")
    f.write(f"\nCLASSIFICATION REPORT:\n")
    f.write(classification_report(labels, predictions, target_names=['Negative', 'Positive']))

print(f"\nResults saved to: {results_file}")
print("="*70)
