#!/usr/bin/env python3
"""
eTripHLApan Training Script
Enhanced TripHLApan with BLOSUM62 encoding and transfer learning
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Change to codes directory (needed for relative paths in helper files)
script_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.join(script_dir, 'codes')
os.chdir(codes_dir)
sys.path.insert(0, codes_dir)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import time

from helper import *
from data_pre_processing2 import *

# Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("="*70)
print("eTripHLApan Training Script")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print(f"CUDA available: {USE_CUDA}")

# Set random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if USE_CUDA:
    torch.cuda.manual_seed(random_seed)

# Hyperparameters
NUM_EPOCHS = 300  # Extended from 100
LEARNING_RATE = 0.0001  # Keep original learning rate
BATCH_SIZE = 512  # Keep original batch size
EARLY_STOPPING_PATIENCE = 30  # Stop if no improvement for 30 epochs

print(f"\nTraining Configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print("="*70)

# Data paths
TRAINING_DATA_PATH = '../for_prediction/training_data.txt'
VALIDATION_DATA_PATH = '../for_prediction/validation_data.txt'
MODEL_SAVE_PATH = '../../models/eTripHLApan/'

# Create model save directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Custom Dataset (from original train.py)
class TripHLApanDataset(Dataset):
    def __init__(self, data_file_path):
        """Load data from TripHLApan format file"""
        self.pep_seq_list = []
        self.allele_seq_list = []
        self.affinity_list = []
        self.label_list = []
        
        # Load allele-to-sequence mapping
        self.dict_allele_seq = map_allele_name_seq()
        
        # Read data file
        with open(data_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    pep = parts[0]
                    hla = parts[1]
                    affinity = float(parts[2])
                    label = int(parts[3])
                    
                    self.pep_seq_list.append(pep)
                    self.allele_seq_list.append(hla)
                    self.affinity_list.append(affinity)
                    self.label_list.append(label)
        
        print(f"Loaded {len(self.pep_seq_list)} samples from {data_file_path}")
        
        # Preprocess sequences
        self._preprocess_sequences()
    
    def _preprocess_sequences(self):
        """Preprocess peptide and allele sequences"""
        processed_peps = []
        processed_alleles = []
        
        # Process peptides - pad to 14 aa
        for pep in self.pep_seq_list:
            peplen = len(pep)
            insert_idx = int((peplen + 1) / 2) - 1
            pseq_pep_seq = pep[0:insert_idx] + 'X' * (14 - peplen) + pep[insert_idx:]
            processed_peps.append(pseq_pep_seq)
        
        # Process alleles
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
        affinity = self.affinity_list[idx]
        label = self.label_list[idx]
        
        # Encode sequences
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


# Network definition (from original helper.py)
class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()

        # network1
        self.gru1 = nn.GRU(20, 128,
                           batch_first=True,
                           bidirectional=True)
        self.gru2 = nn.GRU(20, 128,
                           batch_first=True,
                           bidirectional=True)
        self.full_conn1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network2
        self.embedding1 = nn.Embedding(21, 6)
        self.embedding2 = nn.Embedding(21, 6)

        self.gru3 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True)
        self.gru4 = nn.GRU(6, 128,
                           batch_first=True,
                           bidirectional=True)
        self.full_conn2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # network3
        self.gru5 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True)
        self.gru6 = nn.GRU(28, 128,
                           batch_first=True,
                           bidirectional=True)
        self.full_conn3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
        )

        # Attention layers
        self.attention1 = torch.nn.MultiheadAttention(256, 1)
        self.attention2 = torch.nn.MultiheadAttention(256, 1)
        self.attention3 = torch.nn.MultiheadAttention(256, 1)
        self.attention4 = torch.nn.MultiheadAttention(256, 1)
        self.attention5 = torch.nn.MultiheadAttention(256, 1)
        self.attention6 = torch.nn.MultiheadAttention(256, 1)

        # Final classifier
        self.full_conn = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
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

        result_final = torch.cat((result1, result2, result3), 1)
        output = self.full_conn(result_final)
        return output


# Load data
print("\nLoading dataset...")
train_dataset = TripHLApanDataset(TRAINING_DATA_PATH)
val_dataset = TripHLApanDataset(VALIDATION_DATA_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Initialize model
print("\nInitializing model...")
model = Network_conn().to(DEVICE)

# Load pretrained model weights from original fold1
PRETRAINED_MODEL_PATH = '../../TripHLApan/models/TripHLApan/validate_param_fold1epoch30_batch4154.pkl'
if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
    pretrained_state = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(pretrained_state)
    print("✓ Pretrained weights loaded successfully")
else:
    print(f"⚠ Warning: Pretrained model not found at {PRETRAINED_MODEL_PATH}")
    print("  Training will start from random initialization")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Training loop with early stopping
print("\nStarting training...\n")

best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0
training_log = []
start_time = datetime.now()

for epoch in range(1, NUM_EPOCHS + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        peps1 = batch['pep1'].to(DEVICE)
        alleles1 = batch['allele1'].to(DEVICE)
        peps2 = batch['pep2'].to(DEVICE)
        alleles2 = batch['allele2'].to(DEVICE)
        peps3 = batch['pep3'].to(DEVICE)
        alleles3 = batch['allele3'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        
        outputs = model(peps1, alleles1, peps2, alleles2, peps3, alleles3)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
    
    train_loss /= len(train_loader)
    train_acc = train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            peps1 = batch['pep1'].to(DEVICE)
            alleles1 = batch['allele1'].to(DEVICE)
            peps2 = batch['pep2'].to(DEVICE)
            alleles2 = batch['allele2'].to(DEVICE)
            peps3 = batch['pep3'].to(DEVICE)
            alleles3 = batch['allele3'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(peps1, alleles1, peps2, alleles2, peps3, alleles3)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    # Print progress
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Log metrics
    training_log.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        
        # Save best model
        model_path = os.path.join(MODEL_SAVE_PATH, f'best_model_epoch{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"  → Best model saved at epoch {epoch}")
    else:
        epochs_no_improve += 1
    
    # Early stopping condition
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n⚠ Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
        break

# Training complete
end_time = datetime.now()
elapsed_time = end_time - start_time

print("\n" + "="*70)
print("Training completed!")
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed_time}")
print("="*70)

print(f"\nBest Results:")
print(f"  Best epoch: {best_epoch}")
print(f"  Best validation loss: {best_val_loss:.4f}")

# Save final model
final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.pt')
torch.save(model.state_dict(), final_model_path)

print(f"\nModel saved to:")
print(f"  Best model: {MODEL_SAVE_PATH}/best_model_epoch{best_epoch}.pt")
print(f"  Final model: {final_model_path}")

# Save training log with timestamp to avoid overwriting
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_timestamped = os.path.join(MODEL_SAVE_PATH, f'training_log_{timestamp}.txt')
with open(log_file_timestamped, 'w') as f:
    f.write("="*70 + "\n")
    f.write(f"eTripHLApan Training Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n")
    f.write(f"Initialization: Pretrained model (validate_param_fold1epoch30_batch4154.pkl)\n")
    f.write(f"Encoding Path 1: BLOSUM62 (changed from one-hot)\n")
    f.write(f"Total Epochs: {epoch}\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
    f.write(f"Training Time: {elapsed_time}\n")
    f.write("="*70 + "\n\n")
    f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
    for log in training_log:
        f.write(f"{log['epoch']},{log['train_loss']:.4f},{log['train_acc']:.4f},{log['val_loss']:.4f},{log['val_acc']:.4f}\n")

print(f"  Timestamped log file: {log_file_timestamped}")

print("\n" + "="*70)
