# 📦 Complete Training Package - File Index

## 🎯 What You Have

This package contains everything needed to fine-tune Florence-2 and Qwen2.5-VL models for beaker volume prediction on JarvisLab A6000 GPU.

---

## 📄 Core Files (Must Have)

### 1. `train_vision_models.py` (26 KB)
**Main training script** - The heart of the pipeline
- Loads dataset with streaming (no OOM crashes)
- Trains both Florence-2 and Qwen2.5-VL with LoRA
- Auto-splits data: 70% train, 15% val, 15% test
- Evaluates with MAE, RMSE, R²
- Generates prediction plots
- Optional HuggingFace upload

**Usage:**
```bash
python train_vision_models.py
```

**Before running:**
- Edit line 25: Set `HF_DATASET_NAME`
- Edit line 49: Set `HF_REPO_NAME`

---

### 2. `gradio_demo.py` (15 KB)
**Interactive web demo** for testing trained models
- Upload beaker images
- Get volume predictions
- Compare both models side-by-side
- Custom prompts/questions

**Usage:**
```bash
# Local access
python gradio_demo.py

# Public share link
python gradio_demo.py --share

# Custom model paths
python gradio_demo.py \
  --florence-path ./models/florence2 \
  --qwen-path ./models/qwen2_5vl
```

---

### 3. `requirements.txt` (580 bytes)
**All Python dependencies** needed for training
- PyTorch with CUDA
- Transformers, PEFT, Datasets
- Gradio for demo
- Evaluation libraries

**Usage:**
```bash
pip install -r requirements.txt
```

---

## 🛠️ Setup Files

### 4. `setup.sh` (4.1 KB)
**Automated setup script** for JarvisLab
- Installs system dependencies
- Sets up Python environment
- Installs PyTorch with CUDA
- Verifies GPU
- Prompts for HuggingFace login

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

---

### 5. `run_pipeline.py` (9.8 KB)
**All-in-one automated runner**
- Checks environment
- Verifies dependencies
- Runs training
- Launches demo
- Pretty colored output

**Usage:**
```bash
# Full pipeline
python run_pipeline.py

# Only training
python run_pipeline.py --train-only

# Only demo
python run_pipeline.py --demo-only

# Skip checks
python run_pipeline.py --skip-checks
```

---

## 📚 Documentation Files

### 6. `README.md` (11 KB)
**Complete documentation**
- Feature list
- Step-by-step guide
- Configuration options
- Troubleshooting
- Expected outputs
- Performance tips

**Read this for:** Detailed understanding

---

### 7. `QUICKSTART.md` (4.3 KB)
**5-minute quick start guide**
- Minimal steps to get training
- Timeline expectations
- Success checklist
- Common issues

**Read this for:** Fast setup

---

### 8. `COMMANDS.md` (5.5 KB)
**Command reference cheat sheet**
- All useful commands
- Monitoring commands
- Troubleshooting commands
- Time estimates

**Use this for:** Quick command lookup

---

## 🔧 Optional Files

### 9. `prepare_dataset.py` (14 KB)
**Dataset preparation tool** (if you haven't uploaded to HuggingFace yet)
- Prepares images for HuggingFace
- Supports multiple formats
- Auto-extracts volumes from filenames
- Uploads to HuggingFace Hub

**Usage:**
```bash
# From directory structure
python prepare_dataset.py \
  --mode directory \
  --data-dir ./images \
  --repo-name username/dataset-name

# From JSON labels
python prepare_dataset.py \
  --mode json \
  --data-dir ./images \
  --labels-file labels.json \
  --repo-name username/dataset-name

# From CSV labels
python prepare_dataset.py \
  --mode csv \
  --data-dir ./images \
  --labels-file labels.csv \
  --repo-name username/dataset-name
```

---

## 🚀 Quick Start Workflow

### Option A: Minimal (3 files)
```bash
# 1. Copy these files to JarvisLab:
train_vision_models.py
requirements.txt
gradio_demo.py

# 2. Install
pip install -r requirements.txt

# 3. Configure
nano train_vision_models.py  # Edit dataset name

# 4. Train
python train_vision_models.py

# 5. Demo
python gradio_demo.py --share
```

### Option B: Automated (4 files)
```bash
# 1. Copy these files to JarvisLab:
setup.sh
run_pipeline.py
train_vision_models.py
gradio_demo.py
requirements.txt

# 2. Setup
chmod +x setup.sh
./setup.sh

# 3. Configure
nano train_vision_models.py  # Edit dataset name

# 4. Run pipeline
python run_pipeline.py
```

### Option C: Complete (all 9 files)
Use all files for full documentation and features

---

## 📊 File Dependencies

```
setup.sh
└── requirements.txt

run_pipeline.py
├── train_vision_models.py
└── gradio_demo.py

train_vision_models.py
└── (requires HuggingFace dataset)

gradio_demo.py
└── trained_models/ (output from training)

prepare_dataset.py
└── (optional, for dataset preparation)

Documentation:
├── README.md (comprehensive)
├── QUICKSTART.md (minimal)
└── COMMANDS.md (reference)
```

---

## 💾 Disk Space Requirements

| Item | Size |
|------|------|
| All scripts | <100 KB |
| Dependencies | ~5 GB |
| Florence-2 model | ~900 MB |
| Qwen2.5-VL model | ~8 GB |
| Training checkpoints | ~10 GB |
| Final models | ~9 GB |
| **Total needed** | **~35 GB** |

A6000 has plenty of storage (usually 200+ GB)

---

## 🎯 What Each File Does

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_vision_models.py` | Train models | Always |
| `gradio_demo.py` | Test models | After training |
| `requirements.txt` | Install deps | Setup |
| `setup.sh` | Auto setup | First time |
| `run_pipeline.py` | Automation | Convenience |
| `prepare_dataset.py` | Prep dataset | If needed |
| `README.md` | Learn details | Reference |
| `QUICKSTART.md` | Fast start | Impatient |
| `COMMANDS.md` | Find commands | Ongoing |

---

## ✅ Recommended Workflow

1. **First Time Setup** (10 minutes)
   ```bash
   ./setup.sh
   ```

2. **Configure** (2 minutes)
   ```bash
   nano train_vision_models.py
   # Update dataset name
   ```

3. **Train** (3-5 hours)
   ```bash
   screen -S training
   python train_vision_models.py
   # Ctrl+A, D to detach
   ```

4. **Monitor** (optional)
   ```bash
   tensorboard --logdir=./trained_models --port=6006
   watch -n 1 nvidia-smi
   ```

5. **Test** (anytime after training)
   ```bash
   python gradio_demo.py --share
   ```

---

## 📦 Delivery Checklist

✅ All 9 files included  
✅ Executable permissions set  
✅ Documentation complete  
✅ Examples provided  
✅ Tested on A6000  
✅ Streaming enabled (no OOM)  
✅ LoRA configured  
✅ Auto data split  
✅ Evaluation metrics  
✅ Gradio demo  
✅ HuggingFace integration  

---

## 🎓 Learning Path

1. **Beginner**: Read `QUICKSTART.md` → Run `setup.sh` → Run `run_pipeline.py`
2. **Intermediate**: Read `README.md` → Customize config → Run `train_vision_models.py`
3. **Advanced**: Modify code → Add features → Experiment with hyperparameters

---

## 🔍 Finding Information

| Question | Check |
|----------|-------|
| How to start? | `QUICKSTART.md` |
| What's a command? | `COMMANDS.md` |
| Why error X? | `README.md` → Troubleshooting |
| How to customize? | `README.md` → Configuration |
| What's this file? | This file (`INDEX.md`) |

---

## 🎉 You're Ready!

You now have everything needed to:
- ✅ Fine-tune Florence-2
- ✅ Fine-tune Qwen2.5-VL
- ✅ Use LoRA efficiently
- ✅ Split data automatically
- ✅ Evaluate with metrics
- ✅ Create interactive demos
- ✅ Upload to HuggingFace

**Start with:** `QUICKSTART.md` or `./setup.sh`

**Questions?** Check `README.md` or `COMMANDS.md`

**Good luck with your training! 🚀**

---

## 📞 Support Resources

- **JarvisLab Docs**: https://docs.jarvislabs.ai/
- **HuggingFace Docs**: https://huggingface.co/docs
- **Florence-2**: https://huggingface.co/microsoft/Florence-2-base
- **Qwen2.5-VL**: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- **PEFT (LoRA)**: https://huggingface.co/docs/peft

---

**Package Version:** 1.0  
**Last Updated:** 2026-02-09  
**Optimized For:** JarvisLab A6000 (48GB VRAM)  
**Tested With:** 2000 image dataset
