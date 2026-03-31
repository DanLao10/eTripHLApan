#!/usr/bin/env python3
"""
ROC-AUC Curve Plotting for eTripHLApan Model
Generates a publication-quality ROC curve plot.
"""

import sys
import os

# Change to codes directory for helper imports
script_dir = os.path.dirname(os.path.abspath(__file__))
codes_dir = os.path.join(script_dir, 'eTripHLApan', 'codes')
os.chdir(codes_dir)
sys.path.insert(0, codes_dir)

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helper import *
from data_pre_processing2 import *

# ── Configuration ──────────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

TEST_DATA_PATH = '../for_prediction/test_data.txt'
MODEL_DIR = os.path.join(script_dir, 'models', 'eTripHLApan')
OUTPUT_PATH = os.path.join(script_dir, 'models', 'eTripHLApan', 'roc_auc_best_epoch.pdf')
OUTPUT_PNG = os.path.join(script_dir, 'models', 'eTripHLApan', 'roc_auc_best_epoch.png')
BATCH_SIZE = 512

BEST_EPOCH = 2


# ── Network definition (must match training) ──────────────────────────
class Network_conn(nn.Module):
    def __init__(self):
        super(Network_conn, self).__init__()
        self.gru1 = nn.GRU(20, 128, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(20, 128, batch_first=True, bidirectional=True)
        self.full_conn1 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))

        self.embedding1 = nn.Embedding(21, 6)
        self.embedding2 = nn.Embedding(21, 6)
        self.gru3 = nn.GRU(6, 128, batch_first=True, bidirectional=True)
        self.gru4 = nn.GRU(6, 128, batch_first=True, bidirectional=True)
        self.full_conn2 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))

        self.gru5 = nn.GRU(28, 128, batch_first=True, bidirectional=True)
        self.gru6 = nn.GRU(28, 128, batch_first=True, bidirectional=True)
        self.full_conn3 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))

        self.attention1 = nn.MultiheadAttention(256, 1)
        self.attention2 = nn.MultiheadAttention(256, 1)
        self.attention3 = nn.MultiheadAttention(256, 1)
        self.attention4 = nn.MultiheadAttention(256, 1)
        self.attention5 = nn.MultiheadAttention(256, 1)
        self.attention6 = nn.MultiheadAttention(256, 1)

        self.full_conn = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, peps1, alleles1, peps2, alleles2, peps3, alleles3):
        x1 = self.gru1(peps1)[0]
        q = x1.permute(1, 0, 2)
        x1 = self.attention1(q, q, q)[0].permute(1, 0, 2)[:, -1]

        x2 = self.gru2(alleles1)[0]
        q = x2.permute(1, 0, 2)
        x2 = self.attention2(q, q, q)[0].permute(1, 0, 2)[:, -1]

        result1 = self.full_conn1(torch.cat((x1, x2), 1))

        X1 = self.embedding1(peps2)
        Y1 = self.embedding2(alleles2)

        x1 = self.gru3(X1)[0]
        q = x1.permute(1, 0, 2)
        x1 = self.attention3(q, q, q)[0].permute(1, 0, 2)[:, -1]

        x2 = self.gru4(Y1)[0]
        q = x2.permute(1, 0, 2)
        x2 = self.attention4(q, q, q)[0].permute(1, 0, 2)[:, -1]

        result2 = self.full_conn2(torch.cat((x1, x2), 1))

        x1 = self.gru5(peps3)[0]
        q = x1.permute(1, 0, 2)
        x1 = self.attention5(q, q, q)[0].permute(1, 0, 2)[:, -1]

        x2 = self.gru6(alleles3)[0]
        q = x2.permute(1, 0, 2)
        x2 = self.attention6(q, q, q)[0].permute(1, 0, 2)[:, -1]

        result3 = self.full_conn3(torch.cat((x1, x2), 1))

        return self.full_conn(torch.cat((result1, result2, result3), 1))


# ── Dataset ────────────────────────────────────────────────────────────
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
                    self.pep_seq_list.append(parts[0])
                    self.allele_seq_list.append(parts[1])
                    self.label_list.append(int(parts[3]))

        self._preprocess_sequences()

    def _preprocess_sequences(self):
        processed_peps = []
        for pep in self.pep_seq_list:
            peplen = len(pep)
            idx = int((peplen + 1) / 2) - 1
            processed_peps.append(pep[:idx] + 'X' * (14 - peplen) + pep[idx:])
        processed_alleles = []
        for allele in self.allele_seq_list:
            seq = self.dict_allele_seq.get(allele, 'X' * 200)[:200]
            processed_alleles.append(seq + 'X' * (200 - len(seq)))
        self.pep_seq_list = processed_peps
        self.allele_seq_list = processed_alleles

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        pep = self.pep_seq_list[idx]
        allele = self.allele_seq_list[idx]
        label = self.label_list[idx]
        pep1 = encode_seq(pep, 'BLOSUM62')
        allele1 = encode_seq(allele, 'BLOSUM62')
        pep2 = encode_seq(pep, 'num')
        allele2 = encode_seq(allele, 'num')
        pep3 = encode_seq(pep, 'AAfea_phy')
        allele3 = encode_seq(allele, 'AAfea_phy')
        return {
            'pep1': torch.FloatTensor(pep1), 'allele1': torch.FloatTensor(allele1),
            'pep2': torch.LongTensor(pep2), 'allele2': torch.LongTensor(allele2),
            'pep3': torch.FloatTensor(pep3), 'allele3': torch.FloatTensor(allele3),
            'label': torch.FloatTensor([label]),
        }


# ── Inference helper ───────────────────────────────────────────────────
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            peps1 = batch['pep1'].to(device)
            alleles1 = batch['allele1'].to(device)
            peps2 = batch['pep2'].to(device)
            alleles2 = batch['allele2'].to(device)
            peps3 = batch['pep3'].to(device)
            alleles3 = batch['allele3'].to(device)
            labels = batch['label'].to(device)

            outputs = model(peps1, alleles1, peps2, alleles2, peps3, alleles3)
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))
    return np.array(all_probs), np.array(all_labels)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading test dataset...")
    test_dataset = TripHLApanDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load best model
    path = os.path.join(MODEL_DIR, f'best_model_epoch{BEST_EPOCH}.pt')
    if not os.path.isfile(path):
        print(f"Error: checkpoint not found at {path}")
        return
    print(f"Loading best model (epoch {BEST_EPOCH})...")
    model = Network_conn().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    probs, labels = get_predictions(model, test_loader, DEVICE)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = auc(fpr, tpr)
    print(f"AUC = {auc_val:.4f}")

    # ── Publication-quality plot ───────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 10.5,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # ROC curve
    ax.plot(fpr, tpr, color='#2166ac', lw=2, label=f'ROC curve (AUC = {auc_val:.4f})')

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='grey', lw=1, ls='--', label='Random Classifier (AUC = 0.5)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black',
              framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, ls=':', alpha=0.4)

    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3)
    ax.tick_params(which='major', length=5)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(OUTPUT_PATH)
    print(f"\nPlots saved to:\n  {OUTPUT_PNG}\n  {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == '__main__':
    main()
