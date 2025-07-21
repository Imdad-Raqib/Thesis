# Bengali Sentiment Analysis Model Comparison

## 1. Models Implemented by Us

### Dataset Details
- **Total Samples**: 6,180
- **Class Distribution**: 
  - Negative: 1,977 samples
  - Positive: 2,048 samples
  - Neutral: 2,155 samples
- **Source**: Social media comments (noisy, informal Bengali text)

### Model Performance Summary

| Model Name | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|---------|----------|
| csebuetnlp/banglabert (3-class) | 0.7321 | 0.7297 | 0.7321 | 0.7300 |
| csebuetnlp/banglabert (2-class) | 0.8781 | 0.8793 | 0.8781 | 0.8780 |
| google/muril-base-cased (3-class) | 0.7211 | 0.7229 | 0.7211 | 0.7208 |
| xlm-roberta-base (3-class) | 0.7143 | 0.7137 | 0.7143 | 0.7118 |
| CNN (3-class) | 0.6472 | 0.6470 | 0.6484 | 0.6461 |
| BiLSTM (3-class) | 0.6214 | 0.6238 | 0.6214 | 0.6218 |
| GRU (3-class) | 0.6262 | 0.6278 | 0.6262 | 0.6267 |
| CNN-LSTM (3-class) | 0.6246 | 0.6253 | 0.6246 | 0.6238 |

### Per-Class F1 Scores

| Model | Negative | Positive | Neutral |
|-------|----------|----------|---------|
| csebuetnlp/banglabert | 0.6924 | 0.8187 | 0.6810 |
| google/muril-base-cased | 0.6946 | 0.7873 | 0.6824 |
| xlm-roberta-base | 0.6908 | 0.7818 | 0.6650 |
| CNN | 0.6290 | 0.6981 | 0.6123 |
| BiLSTM | 0.6247 | 0.6420 | 0.6000 |
| GRU | 0.6044 | 0.6783 | 0.5981 |
| CNN-LSTM | 0.6235 | 0.6667 | 0.5833 |

### Detailed Model Descriptions

#### Model 1: csebuetnlp/banglabert
- **Architecture**: BanglaBERT with custom classifier head
- **Configuration**: max_length=128, epochs=7, lr=2e-5, batch_size=32, dropout=0.3
- **Key Findings**: Best transformer performance. 3-class: 73.21% accuracy. 2-class: 87.81% accuracy

#### Model 2: google/muril-base-cased
- **Architecture**: MURIL (Multilingual Indian Language model)
- **Configuration**: Same as BanglaBERT (epochs=7, lr=2e-5, batch_size=32)
- **Key Findings**: 72.11% accuracy, slightly lower than BanglaBERT

#### Model 3: XLM-RoBERTa
- **Architecture**: Multilingual RoBERTa model
- **Configuration**: 
  - Model: xlm-roberta-base
  - Max length: 128, epochs: 7, lr: 2e-5, batch_size: 32
  - Same training strategy as BanglaBERT
- **Test Performance**: 
  - Accuracy: 71.43%, F1: 71.18%, F1 Macro: 71.25%
  - Loss: 0.8379
  - Strong performance on Positive class (F1: 0.7818)
  - Good generalization (CV: 71.95%, Test: 71.43%, Difference: -0.52%)
  - High confidence predictions: 39.2% with >0.9 confidence

#### Model 4: CNN
- **Architecture**: 1D Convolutional Neural Network
- **Configuration**: 
  - Filter sizes: [3, 4, 5] with 100 filters each
  - Embedding dimension: 100
  - Epochs: 25, batch_size: 64, lr: 0.001
- **Test Performance**: 
  - Accuracy: 64.72%, F1: 64.61%, Loss: 1.3903
  - Best performing non-transformer model
  - Strong performance on Positive class (F1: 0.6981)

#### Model 4: BiLSTM
- **Architecture**: Bidirectional LSTM (without attention)
- **Configuration**: 
  - Hidden dimension: 128
  - 2 LSTM layers
  - Epochs: 25, batch_size: 64
- **Test Performance**: 
  - Accuracy: 62.14%, F1: 62.18%, Loss: 1.7007
  - Balanced performance across classes
  - Relatively higher loss compared to CNN

#### Model 5: GRU
- **Architecture**: Gated Recurrent Unit
- **Configuration**: 
  - Hidden dimension: 128
  - 2 GRU layers
  - Similar training config as BiLSTM
- **Test Performance**: 
  - Accuracy: 62.62%, F1: 62.67%, Loss: 1.5953
  - Better than BiLSTM but lower than CNN
  - Best performance on Positive class (F1: 0.6783)

#### Model 6: CNN-LSTM
- **Architecture**: Hybrid CNN followed by LSTM
- **Configuration**: 
  - CNN for feature extraction
  - LSTM for sequence modeling
  - Combined architecture
- **Test Performance**: 
  - Accuracy: 62.46%, F1: 62.38%, Loss: 2.0577
  - Highest loss among all models
  - Underperformed compared to standalone CNN

### Methodology for Best Performing Model (BanglaBERT)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        METHODOLOGY PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Raw Bengali Text  │     │  Data Preprocessing │     │ Cleaned Dataset     │
│ (Social Media)      │ --> │ • Remove URLs       │ --> │ • No duplicates     │
│ • Noisy            │     │ • Clean Special Chars│     │ • Min 3 words       │
│ • Informal         │     │ • Handle Punctuation │     │ • Length filtered   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                                                    │
                                                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Augmentation (Training Only)               │
├─────────────────────────────────────────────────────────────────────┤
│  • Punctuation variation (30% probability)                          │
│  • Word repetition for emphasis                                     │
│  • Middle word shuffling (preserving start/end)                     │
└─────────────────────────────────────────────────────────────────────┘
                                                                    │
                                                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Transformer Model Architecture (Universal)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Text ──> [Model Tokenizer] ──> Token IDs (max_len=128)      │
│                                         │                            │
│                                         ▼                            │
│                    ┌────────────────────────────────┐               │
│                    │   Pre-trained Transformer      │               │
│                    │   • BanglaBERT                 │               │
│                    │   • MURIL                      │               │
│                    │   • XLM-RoBERTa                │               │
│                    │   Hidden Size: 768             │               │
│                    └────────────────────────────────┘               │
│                                         │                            │
│                                         ▼                            │
│                    ┌────────────────────────────────┐               │
│                    │   Enhanced Classifier Head     │               │
│                    ├────────────────────────────────┤               │
│                    │   Layer Norm (768)             │               │
│                    │           ↓                    │               │
│                    │   Dropout (0.3)                │               │
│                    │           ↓                    │               │
│                    │   Pre-classifier (768 → 768)   │               │
│                    │           ↓                    │               │
│                    │   GELU Activation              │               │
│                    │           ↓                    │               │
│                    │   Classifier (768 → 384)       │               │
│                    │           ↓                    │               │
│                    │   GELU Activation              │               │
│                    │           ↓                    │               │
│                    │   Layer Norm (384)             │               │
│                    │           ↓                    │               │
│                    │   Dropout (0.15)               │               │
│                    │           ↓                    │               │
│                    │   Output Layer (384 → 3/2)     │               │
│                    └────────────────────────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Strategy                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  • Optimizer: AdamW with Layer-wise Learning Rates                  │
│    - Transformer layers: 1e-5                                       │
│    - Classifier layers: 2e-5                                        │
│    - Adam epsilon: 1e-8                                             │
│    - Weight decay: 0.01                                             │
│                                                                      │
│  • Scheduler: Linear Warmup (10% steps) → Linear Decay              │
│  • Loss: CrossEntropyLoss with Label Smoothing (0.1)                │
│  • Epochs: 7 (with early stopping, patience=7)                      │
│  • Batch Size: 32                                                   │
│  • Gradient Clipping: 1.0                                           │
│  • Mixed Precision: FP16 enabled                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Evaluation                                 │
├─────────────────────────────────────────────────────────────────────┤
│  • 5-Fold Stratified Cross Validation                               │
│  • Data Split: 60% train, 20% val, 20% test (per fold)             │
│  • Metrics: Accuracy, Precision, Recall, F1-Score                   │
│  • Per-class and Weighted Metrics                                   │
│  • Early Stopping based on validation F1-score                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Methodology for Deep Learning Models (CNN, BiLSTM, GRU, CNN-LSTM)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEEP LEARNING MODELS PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Raw Bengali Text  │     │  Text Preprocessing │     │ Cleaned Text        │
│ (Social Media)      │ --> │ • Remove special    │ --> │ • Bengali chars     │
│ • 6,180 samples    │     │   chars (keep       │     │ • Numbers           │
│ • 3 classes        │     │   Bengali Unicode)  │     │ • Basic punctuation │
└─────────────────────┘     │ • Normalize spaces  │     └─────────────────────┘
                            └─────────────────────┘                    │
                                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Vocabulary Building                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Tokenization: Simple split-based                                 │
│  • Min word frequency: 2                                            │
│  • Special tokens: <PAD>, <UNK>, <SOS>, <EOS>                      │
│  • Vocabulary size: ~10,000-15,000 words                           │
│  • Max sequence length: 100 tokens                                  │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Embedding Initialization                        │
├─────────────────────────────────────────────────────────────────────┤
│  • Method: Random initialization                                     │
│  • Embedding dimension: 100                                          │
│  • Special handling: <PAD> token → zero vector                      │
│  • Frequency-based adjustments:                                     │
│    - High frequency words (>10): embeddings × 1.1                  │
│    - Low frequency words (<3): embeddings × 0.9                    │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                          │
                    ▼                                          ▼
┌───────────────────────────────┐          ┌──────────────────────────────┐
│         CNN Model             │          │      BiLSTM Model            │
├───────────────────────────────┤          ├──────────────────────────────┤
│ • Filter sizes: [3, 4, 5]     │          │ • Hidden dim: 128            │
│ • Filters per size: 100       │          │ • Layers: 2                  │
│ • Total filters: 300          │          │ • Bidirectional: Yes         │
│ • Conv1D + BatchNorm          │          │ • Attention mechanism        │
│ • Max pooling over time       │          │ • Pack padded sequences      │
│ • FC layers: 300→128→3        │          │ • FC layers: 256→128→3       │
│ • Dropout: 0.5                │          │ • Dropout: 0.5               │
└───────────────────────────────┘          └──────────────────────────────┘
                    │                                          │
                    ▼                                          ▼
┌───────────────────────────────┐          ┌──────────────────────────────┐
│         GRU Model             │          │    CNN-LSTM Hybrid           │
├───────────────────────────────┤          ├──────────────────────────────┤
│ • Hidden dim: 128             │          │ CNN Component:               │
│ • Layers: 2                   │          │ • Same as CNN model          │
│ • Bidirectional: Yes          │          │ LSTM Component:              │
│ • Attention mechanism         │          │ • Input: CNN features        │
│ • Similar to BiLSTM           │          │ • Hidden dim: 128            │
│ • FC layers: 256→128→3        │          │ • Layers: 2                  │
│ • Dropout: 0.5                │          │ • Attention on LSTM output   │
└───────────────────────────────┘          └──────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Training Configuration                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Optimizer: Adam (lr=0.001, weight_decay=1e-5)                   │
│  • Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)           │
│  • Loss: CrossEntropyLoss                                          │
│  • Batch size: 64                                                  │
│  • Epochs: 25 (with early stopping, patience=5)                    │
│  • Gradient clipping: max_norm=1.0                                 │
│  • Evaluation metric: Weighted F1-score                            │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Splitting                               │
├─────────────────────────────────────────────────────────────────────┤
│  • Train: 72% (~4,450 samples)                                     │
│  • Validation: 18% (~1,112 samples)                                │
│  • Test: 10% (~618 samples)                                        │
│  • Stratified splits to maintain class distribution                 │
│  • Random seed: 42 for reproducibility                             │
└─────────────────────────────────────────────────────────────────────┘
```

#### Data Preprocessing Details
1. **Text Cleaning**: 
   - Removed URLs and special characters
   - Normalized punctuation variations (।!|.? → single space)
   - Removed multiple spaces
   - Filtered texts with less than 3 words
   - Removed duplicates based on normalized text
   - Removed length outliers (1st and 99th percentile)

2. **Data Quality Checks**:
   - Identified ~1000 highly similar text pairs (>90% similarity)
   - Vocabulary analysis: Total unique words identified
   - Class-specific text length and diversity analysis

3. **No Class Balancing**: 
   - Original class distribution maintained
   - No SMOTE or other balancing techniques applied

#### Model Architecture Details
1. **Base Model**: Pre-trained csebuetnlp/banglabert
2. **Custom Classifier Head**:
   - Layer normalization on BERT output (hidden_size=768)
   - Dropout (p=0.3)
   - Pre-classifier dense layer (768 → 768)
   - GELU activation
   - Classifier dense layer (768 → 384)
   - GELU activation
   - Layer normalization (384 dimensions)
   - Dropout (p=0.15)
   - Output layer (384 → num_classes)
   - Weight initialization: Normal distribution (mean=0, std=0.02)

#### Training Strategy Details
1. **Optimizer Configuration**:
   - AdamW optimizer with differential learning rates
   - BERT parameters: lr = 1e-5
   - Classifier parameters: lr = 2e-5
   - Weight decay: 0.01
   - Adam epsilon: 1e-8

2. **Learning Rate Schedule**:
   - Linear warmup for 10% of total training steps
   - Linear decay after warmup
   - Total training steps = num_epochs × len(train_loader)

3. **Regularization**:
   - Label smoothing factor: 0.1
   - Gradient clipping: max_norm = 1.0
   - Dropout rates: 0.3 (first layer), 0.15 (second layer)
   - Layer normalization after activations

4. **Data Augmentation** (30% probability during training):
   - Punctuation variations (random punctuation at end)
   - Word repetition (emphasize random words)
   - Middle word shuffling (preserve first/last words)

5. **Early Stopping**:
   - Monitor validation F1-score
   - Patience: 7 epochs
   - Threshold: 0.0001 improvement required
   - Best model checkpoint saved based on F1-score

#### Evaluation Methodology
1. **5-Fold Stratified Cross-Validation**:
   - Stratified splits to maintain class distribution
   - Each fold: ~80% train, ~20% validation
   - Final test set: Separate 20% holdout

2. **Metrics Tracked**:
   - Accuracy, Precision, Recall, F1-score (weighted)
   - Per-class F1-scores
   - Loss values (train and validation)
   - Confidence score distributions
   - Calibration analysis

3. **Error Analysis**:
   - False positive/negative breakdown
   - High-confidence error identification
   - Text length impact on errors
   - Sample misclassified texts analysis

## 2. Comparison with Models from Literature

### 3-Class Classification Comparison

| Paper | Model/Approach | Dataset Details | Test Accuracy | Test F1-Score |
|-------|----------------|-----------------|---------------|---------------|
| **Our Best Model** | BanglaBERT with custom classifier | 6,180 social media comments | **0.7321** | **0.7300** |
| Alam et al. (2020) | XLM-RoBERTa-large | 8,910 YouTube comments | 0.7411 | 0.7407 |
| Fahima Hossain et al. (2025) | LSTM with hyperparameter tuning | ~3,000 social media comments | 0.6500 | 0.6500 |



### 2-Class Classification Comparison

| Paper | Model/Approach | Test Accuracy | Test F1-Score |
|-------|----------------|---------------|---------------|
| Our Best Model | BanglaBERT | 0.8781 | 0.8780 |
| Zishan Ahmed et al. (2023) | Bangla-BERT | 0.7806 | 0.7755 |
| Mahmudul Hasan et al. (2024) | Logistic Regression (Best) | 0.7478 | 0.7477 |

**Performance Gap**: Our BanglaBERT outperforms best literature model by **+9.75%**

**Zishan Ahmed et al. Dataset**: ~1,000 e-commerce reviews from Daraz, 2-class and 4-class versions, formal product reviews

**Mahmudul Hasan et al. Dataset**: ABSA datasets with complex/compound sentences, 2 aspects per sentence, domains: Restaurant (801), Movie (800), Mobile (975), Car (1149)