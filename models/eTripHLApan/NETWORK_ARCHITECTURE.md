# eTripHLApan Network Architecture

## Overview
eTripHLApan implements a **Triple Coding Matrix** approach with 3 parallel neural network paths that process peptide and HLA allele sequences using different encodings, then fuses the learned representations.

**Total Parameters:** 2,507,645

---

## Network Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                 │
├──────────────────┬──────────────────────────────────────────────────┤
│  Peptide (8-14aa)│            HLA Allele (200 aa)                   │
└────────┬─────────┴─────────────────────────────────────────────────┬┘
         │                                                             │
         │            ┌──────────────────────────────┐               │
         │            │  Distributed to 3 Paths      │               │
         │            └──────────────────────────────┘               │
         │                                                             │
         ▼                  ▼                  ▼
┌────────────────────┐┌────────────────────┐┌────────────────────┐
│   PATH 1           ││   PATH 2           ││   PATH 3           │
│ One-hot Encoding   ││ Embedding +        ││ Physicochemical    │
│  (20 dimensions)   ││ Numeric Encoding   ││ Encoding           │
│                    ││  (6 dimensions)    ││ (28 dimensions)    │
├────────────────────┤├────────────────────┤├────────────────────┤
│  GRU1 (Peptide)    ││  GRU3 (Peptide)    ││  GRU5 (Peptide)    │
│  20 → 128×2        ││  6 → 128×2         ││  28 → 128×2        │
├────────────────────┤├────────────────────┤├────────────────────┤
│ MultiheadAttention1││ MultiheadAttention3││ MultiheadAttention5│
│  (256 dimensions)  ││  (256 dimensions)  ││  (256 dimensions)  │
├────────────────────┤├────────────────────┤├────────────────────┤
│  GRU2 (HLA)        ││  GRU4 (HLA)        ││  GRU6 (HLA)        │
│  20 → 128×2        ││  6 → 128×2         ││  28 → 128×2        │
├────────────────────┤├────────────────────┤├────────────────────┤
│ MultiheadAttention2││ MultiheadAttention4││ MultiheadAttention6│
│  (256 dimensions)  ││  (256 dimensions)  ││  (256 dimensions)  │
├────────────────────┤├────────────────────┤├────────────────────┤
│  Concatenate       ││  Concatenate       ││  Concatenate       │
│  (512 dimensions)  ││  (512 dimensions)  ││  (512 dimensions)  │
├────────────────────┤├────────────────────┤├────────────────────┤
│  FC1: 512 → 128    ││  FC2: 512 → 128    ││  FC3: 512 → 128    │
│  (128 dimensions)  ││  (128 dimensions)  ││  (128 dimensions)  │
└────────┬───────────┘└────────┬───────────┘└────────┬───────────┘
         │                     │                     │
         └─────────────┬───────┴────────┬────────────┘
                       ▼
         ┌─────────────────────────────┐
         │  FUSION LAYER               │
         │  Concatenate 3 Paths        │
         │  (384 dimensions)           │
         └────────────┬────────────────┘
                      ▼
         ┌──────────────────────────┐
         │  FC: 384 → 128           │
         │  Fully Connected Layer   │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  ReLU Activation         │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  FC: 128 → 128           │
         │  Fully Connected Layer   │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  ReLU Activation         │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  Dropout (p=0.2)         │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  FC: 128 → 1             │
         │  Output Layer            │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │  Sigmoid Activation      │
         │  Output: [0, 1]          │
         │ (Binding Probability)    │
         └──────────────────────────┘
```

---

## Detailed Component Breakdown

### INPUT
- **Peptide Sequence:** 8-14 amino acids
- **HLA Allele Sequence:** ~200 amino acids
- **Batch Processing:** Sequences are padded/truncated to fixed lengths

### PATH 1: One-Hot Encoding
Represents each amino acid as a 20-dimensional binary vector (one-hot)
- **Encoding Dimension:** 20
- **GRU1 (Peptide):** BiGRU with 128 hidden units → 256-dim output
- **Attention1:** MultiheadAttention → 256 dimensions
- **GRU2 (HLA):** BiGRU with 128 hidden units → 256-dim output
- **Attention2:** MultiheadAttention → 256 dimensions
- **Concat:** GRU + Attention outputs → 512 dimensions
- **FC1:** Fully Connected 512 → 128 dimensions

### PATH 2: Embedding + Numeric Encoding
Combines learned embeddings with numeric physicochemical properties
- **Encoding Dimension:** 6 (includes amino acid embedding + numeric features)
- **GRU3 (Peptide):** BiGRU with 128 hidden units → 256-dim output
- **Attention3:** MultiheadAttention → 256 dimensions
- **GRU4 (HLA):** BiGRU with 128 hidden units → 256-dim output
- **Attention4:** MultiheadAttention → 256 dimensions
- **Concat:** GRU + Attention outputs → 512 dimensions
- **FC2:** Fully Connected 512 → 128 dimensions

### PATH 3: Physicochemical Encoding
Uses 28-dimensional physicochemical property vectors (BLOSUM scores, etc.)
- **Encoding Dimension:** 28 (physicochemical features)
- **GRU5 (Peptide):** BiGRU with 128 hidden units → 256-dim output
- **Attention5:** MultiheadAttention → 256 dimensions
- **GRU6 (HLA):** BiGRU with 128 hidden units → 256-dim output
- **Attention6:** MultiheadAttention → 256 dimensions
- **Concat:** GRU + Attention outputs → 512 dimensions
- **FC3:** Fully Connected 512 → 128 dimensions

### FUSION LAYER
- **Input:** 3 × 128-dim = 384 dimensions
- **Operation:** Concatenate outputs from all 3 paths
- **Output:** 384 dimensions

### CLASSIFICATION HEAD
- **FC1:** 384 → 128 dimensions
- **Activation:** ReLU
- **FC2:** 128 → 128 dimensions
- **Activation:** ReLU
- **Regularization:** Dropout (p=0.2)
- **FC3:** 128 → 1 dimension
- **Activation:** Sigmoid
- **Output:** Binding probability [0, 1]

---

## Key Features

1. **Triple Encoding Perspective**
   - One-hot: Standard amino acid representation
   - Embedding: Learned latent space representation
   - Physicochemical: Domain knowledge-based features

2. **Bidirectional GRU Processing**
   - Captures both forward and backward sequence information
   - 6 GRU layers total (2 per path × 3 paths)
   - Hidden size: 128 (becomes 256 after bidirectional)

3. **Multi-Head Attention Mechanism**
   - 6 attention layers total (2 per path × 3 paths)
   - Learns to focus on important amino acid positions
   - Integrates information across sequence positions

4. **Early Fusion with Late Integration**
   - Each path independently processes and learns representations
   - Representations fused at 384-dim layer
   - Final classification performed on fused features

5. **Regularization**
   - Dropout (p=0.2) in classification head
   - No explicit L1/L2 regularization in training
   - Early stopping on validation loss

---

## Hyperparameter Summary

| Component | Value |
|-----------|-------|
| Peptide Max Length | 14 |
| HLA Max Length | 200 |
| GRU Hidden Units | 128 |
| Attention Heads | 1 |
| Path 1 Input Dim | 20 (one-hot) |
| Path 2 Input Dim | 6 (embedding) |
| Path 3 Input Dim | 28 (physicochemical) |
| Fusion Dimension | 384 (128×3) |
| Classification Hidden | 128 |
| Dropout Rate | 0.2 |
| Total Parameters | 2,507,645 |

---

## Data Flow Summary

```
Input: (batch_size, seq_len, encoding_dim)
  ↓
Path 1-3 (Parallel Processing):
  - Encoding → GRU → Attention → Concat → FC
  - Output: (batch_size, 128)
  ↓
Fusion:
  - Concatenate 3 paths: (batch_size, 384)
  ↓
Classification:
  - FC → ReLU → FC → ReLU → Dropout → FC → Sigmoid
  - Output: (batch_size, 1) ∈ [0, 1]
```

---

## Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 0.0001 (fixed, no scheduling)
- **Loss Function:** BCELoss (Binary Cross-Entropy)
- **Batch Size:** 512
- **Max Epochs:** 300
- **Early Stopping Patience:** 30 epochs
- **Early Stopping Metric:** Validation Loss

---

## Model Performance (eTripHLApan)

| Metric | Value |
|--------|-------|
| Test Accuracy | 76.92% |
| Test AUC | 0.8527 |
| Precision | 79.87% |
| Recall | 83.22% |
| F1-Score | 81.51% |
| Specificity | 67.03% |
| Training Epochs | 104 (stopped by early stopping) |
| Best Epoch | 69 |

---

## Use Cases

This architecture is designed for **HLA-I peptide binding prediction**, predicting whether a peptide sequence will bind to a given HLA allele with high affinity. The triple-path design allows the model to learn complementary representations:

- **One-hot path:** Learns basic amino acid identities
- **Embedding path:** Learns latent features from patterns in data
- **Physicochemical path:** Leverages domain knowledge about amino acid properties

The fusion of these three perspectives improves prediction accuracy compared to using a single encoding method.

