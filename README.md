# 🧪 Beaker Volume Prediction - Florence-2 & Qwen2.5-VL

Complete training pipeline for fine-tuning Florence-2 and Qwen2.5-VL models on beaker volume prediction using LoRA on JarvisLab A6000 GPU.

## 📋 Features

✅ **Dual Model Training**: Florence-2 Base + Qwen2.5-VL 2B  
✅ **LoRA Fine-tuning**: Efficient training with low memory footprint  
✅ **Streaming Dataset**: Prevents OOM crashes on large datasets  
✅ **Automatic Data Split**: 70% train / 15% validation / 15% test  
✅ **Comprehensive Evaluation**: MAE, RMSE, R² metrics  
✅ **Visualization**: Prediction plots and error distributions  
✅ **Gradio Demo**: Interactive web interface for testing  
✅ **HuggingFace Integration**: Optional model upload  
✅ **Optimized for A6000**: Configured for 48GB VRAM  

## 🚀 Quick Start (JarvisLab)

### Step 1: Clone Repository

```bash
# SSH into your JarvisLab instance
cd ~
mkdir beaker-volume-training
cd beaker-volume-training

# Copy all files to this directory
# (train_vision_models.py, gradio_demo.py, requirements.txt, setup.sh)
```

### Step 2: Run Setup Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

This will:
- Install system dependencies
- Set up Python environment
- Install PyTorch with CUDA
- Install all required packages
- Verify GPU availability
- Prompt for HuggingFace login

### Step 3: Configure Your Dataset

Edit `train_vision_models.py` and update:

```python
# Line ~25
HF_DATASET_NAME = "your-username/beaker-volume-dataset"

# Line ~49  
HF_REPO_NAME = "your-username/beaker-volume-model"

# Line ~47 (optional)
UPLOAD_TO_HF = True  # Set to True to upload after training
```

### Step 4: Start Training

```bash
# Start training (will take 2-4 hours depending on dataset size)
python train_vision_models.py
```

### Step 5: Monitor Training (Optional)

In a separate terminal:

```bash
# Start TensorBoard
tensorboard --logdir=./trained_models --port=6006

# Access at: http://localhost:6006
# Or if using JarvisLab, use port forwarding
```

### Step 6: Launch Demo

After training completes:

```bash
# Launch Gradio demo
python gradio_demo.py --share

# Access the public URL that appears in the terminal
```

## 📁 Project Structure

```
beaker-volume-training/
├── train_vision_models.py   # Main training script
├── gradio_demo.py           # Interactive demo interface
├── requirements.txt         # Python dependencies
├── setup.sh                 # Automated setup script
├── README.md               # This file
└── trained_models/         # Output directory (created during training)
    ├── florence2_final/
    ├── qwen2_5vl_final/
    ├── evaluation_results.json
    ├── Florence-2_evaluation.png
    └── Qwen2.5-VL_evaluation.png
```

## 🔧 Configuration Options

### Dataset Configuration

```python
class Config:
    # Your HuggingFace dataset
    HF_DATASET_NAME = "username/dataset-name"
    
    # Enable streaming to avoid OOM
    STREAMING = True
    
    # Data split ratios
    TRAIN_SPLIT = 0.70  # 70%
    VAL_SPLIT = 0.15    # 15%
    TEST_SPLIT = 0.15   # 15%
```

### Model Selection

```python
    # Choose model sizes
    FLORENCE_MODEL = "microsoft/Florence-2-base"      # or Florence-2-large
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"         # 2B is smallest
```

### LoRA Parameters

```python
    # LoRA configuration
    LORA_R = 8              # Rank (higher = more parameters)
    LORA_ALPHA = 16         # Scaling factor
    LORA_DROPOUT = 0.05     # Dropout rate
```

### Training Hyperparameters

```python
    # Training settings
    BATCH_SIZE = 4                    # Adjust based on VRAM
    GRADIENT_ACCUMULATION = 4         # Effective batch = 4 × 4 = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    MAX_LENGTH = 512
```

## 📊 Expected Output

### During Training

```
================================================================================
🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL
================================================================================

📥 Loading dataset with streaming...
📊 Creating train/val/test splits with streaming...
Processed 100 examples - Train: 71, Val: 14, Test: 15
Processed 200 examples - Train: 142, Val: 28, Test: 30
...

✅ Dataset split complete:
   Train: 1400 examples
   Val: 300 examples
   Test: 300 examples

================================================================================
FLORENCE-2 TRAINING
================================================================================

🤖 Setting up Florence-2 model: microsoft/Florence-2-base
trainable params: 2,359,296 || all params: 232,359,296 || trainable%: 1.0156

🚀 Starting Florence-2 training...
Epoch 1/10: 100%|██████████| 350/350 [12:34<00:00,  2.15s/it, loss=0.234]
Epoch 2/10: 100%|██████████| 350/350 [12:31<00:00,  2.14s/it, loss=0.156]
...

✅ Florence-2 training complete! Model saved to ./trained_models/florence2_final

================================================================================
QWEN2.5-VL TRAINING
================================================================================

🤖 Setting up Qwen2.5-VL model: Qwen/Qwen2-VL-2B-Instruct
trainable params: 3,145,728 || all params: 2,003,145,728 || trainable%: 0.1570

🚀 Starting Qwen2.5-VL training...
Epoch 1/10: 100%|██████████| 350/350 [15:23<00:00,  2.64s/it, loss=0.198]
...

================================================================================
MODEL EVALUATION
================================================================================

📊 Evaluating Florence-2...

📈 Florence-2 Results:
   MAE:  12.34 mL
   RMSE: 18.67 mL
   R²:   0.9234
   Plot saved to: ./trained_models/Florence-2_evaluation.png

📊 Evaluating Qwen2.5-VL...

📈 Qwen2.5-VL Results:
   MAE:  10.87 mL
   RMSE: 15.92 mL
   R²:   0.9456
   Plot saved to: ./trained_models/Qwen2.5-VL_evaluation.png

✅ Results saved to: ./trained_models/evaluation_results.json

================================================================================
🎉 TRAINING COMPLETE!
================================================================================

Models saved to:
  Florence-2:   ./trained_models/florence2_final
  Qwen2.5-VL:   ./trained_models/qwen2_5vl_final

Results:       ./trained_models/evaluation_results.json
```

### Evaluation Results File

```json
{
  "florence2": {
    "mae": 12.34,
    "rmse": 18.67,
    "r2": 0.9234,
    "predictions": [250.0, 125.5, 380.2, ...],
    "ground_truth": [250.0, 130.0, 375.0, ...]
  },
  "qwen2_5vl": {
    "mae": 10.87,
    "rmse": 15.92,
    "r2": 0.9456,
    "predictions": [251.2, 128.3, 377.8, ...],
    "ground_truth": [250.0, 130.0, 375.0, ...]
  },
  "config": {
    "train_size": 1400,
    "val_size": 300,
    "test_size": 300,
    "epochs": 10,
    "batch_size": 4,
    "learning_rate": 0.0002
  }
}
```

## 🎨 Gradio Demo Features

The demo provides three tabs:

1. **Florence-2 Tab**: Test Florence-2 model individually
2. **Qwen2.5-VL Tab**: Test Qwen2.5-VL model individually  
3. **Compare Models Tab**: Side-by-side comparison

Each interface allows you to:
- Upload beaker images
- Customize prompts/questions
- View full model responses
- See extracted volume values

## 💾 Memory Requirements

### Model Sizes

| Model | Base Size | LoRA Size | Total VRAM |
|-------|-----------|-----------|------------|
| Florence-2 Base | 230M | 2.4M | ~2 GB |
| Qwen2.5-VL 2B | 2B | 3.1M | ~8 GB |

### Training Memory

- **Batch Size 4**: ~12 GB VRAM per model
- **Gradient Accumulation 4**: Effective batch of 16
- **Total Peak**: ~15 GB (well within A6000's 48GB)

## 📤 Uploading to HuggingFace

To upload your trained models:

1. Set `UPLOAD_TO_HF = True` in config
2. Update `HF_REPO_NAME` with your desired repository name
3. Make sure you're logged in: `huggingface-cli login`

Models will be uploaded to:
- `username/repo-name-florence2`
- `username/repo-name-qwen2-5vl`

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
BATCH_SIZE = 2

# Increase gradient accumulation
GRADIENT_ACCUMULATION = 8
```

### Dataset Loading Issues

```python
# If streaming fails, disable it
STREAMING = False

# Note: This will load entire dataset into memory
```

### Model Download Issues

```bash
# If model download is slow or fails
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf-transfer
```

### CUDA Out of Memory

```python
# Enable gradient checkpointing
self.model.gradient_checkpointing_enable()

# Use 8-bit training
load_in_8bit=True
```

## 📈 Performance Tips

1. **Faster Training**: Enable flash attention (requires compatible GPU)
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Better Accuracy**: Increase epochs or learning rate
   ```python
   NUM_EPOCHS = 15
   LEARNING_RATE = 3e-4
   ```

3. **Reduce Overfitting**: Increase dropout
   ```python
   LORA_DROPOUT = 0.1
   ```

4. **Monitor Training**: Use TensorBoard
   ```bash
   tensorboard --logdir=./trained_models
   ```

## 📝 Dataset Format

Your HuggingFace dataset should have:

```python
{
    'image': PIL.Image or image path,
    'answer': "The volume is 250 mL",  # or
    'volume': "250 mL"                  # Either field works
}
```

Example dataset structure:
```
beaker-volume-dataset/
├── data/
│   ├── normal_background/
│   │   ├── beaker_001.jpg
│   │   ├── beaker_002.jpg
│   │   └── ...
│   └── cluttered_background/
│       ├── beaker_001.jpg
│       └── ...
└── metadata.jsonl
```

## 🔍 Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average error in mL
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **R² (R-squared)**: How well predictions fit ground truth (0-1, higher is better)

## 🎯 Expected Performance

With 2000 training examples:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Florence-2 | 10-15 mL | 15-20 mL | 0.90-0.95 |
| Qwen2.5-VL | 8-12 mL | 12-18 mL | 0.92-0.96 |

## 📚 Additional Resources

- [Florence-2 Paper](https://arxiv.org/abs/2311.06242)
- [Qwen2-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review error messages in terminal
3. Check TensorBoard for training curves
4. Verify GPU availability: `nvidia-smi`

## 📄 License

This project uses:
- Florence-2: MIT License
- Qwen2.5-VL: Apache 2.0 License
- LoRA (PEFT): Apache 2.0 License

## 🙏 Acknowledgments

- Microsoft for Florence-2
- Alibaba for Qwen2.5-VL
- HuggingFace for transformers and PEFT
- JarvisLab for GPU infrastructure
