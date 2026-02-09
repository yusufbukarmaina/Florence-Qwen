# ⚡ Quick Start Guide - 5 Minutes to Training

## 🎯 For Busy People

Just want to start training immediately? Follow these steps:

### Step 1: Copy Files (30 seconds)
```bash
# SSH into JarvisLab
cd ~
mkdir beaker-training && cd beaker-training

# Copy all provided files here:
# - train_vision_models.py
# - gradio_demo.py
# - requirements.txt
# - setup.sh
# - prepare_dataset.py (if needed)
```

### Step 2: Install (5 minutes)
```bash
chmod +x setup.sh
./setup.sh
```

When prompted for HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create new token
3. Paste it

### Step 3: Configure (1 minute)
```bash
nano train_vision_models.py

# Change these lines (around line 25 and 49):
HF_DATASET_NAME = "your-username/your-dataset-name"
HF_REPO_NAME = "your-username/your-model-name"

# Save: Ctrl+O, Enter, Ctrl+X
```

### Step 4: Train (3-5 hours)
```bash
# Use screen to keep running if connection drops
screen -S training

python train_vision_models.py

# Detach: Ctrl+A then D
# Reattach later: screen -r training
```

### Step 5: Demo
```bash
python gradio_demo.py --share

# Click the public URL that appears
```

## ✅ Done!

Your models are training. While waiting:
- Monitor: `nvidia-smi`
- TensorBoard: `tensorboard --logdir=./trained_models --port=6006`
- Check logs: `tail -f training.log`

---

## 🆘 Troubleshooting

### "No module named X"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Edit `train_vision_models.py`:
```python
BATCH_SIZE = 2  # Reduce from 4
```

### "Dataset not found"
Make sure your dataset is public or you're logged in:
```bash
huggingface-cli login
```

### "Training too slow"
Check GPU usage:
```bash
nvidia-smi
# Should show ~90%+ GPU utilization
```

---

## 📊 Expected Timeline

| Task | Time | GPU % |
|------|------|-------|
| Setup | 5 min | 0% |
| Dataset load | 2 min | 0% |
| Florence-2 train | 1.5 hr | 95% |
| Qwen2.5-VL train | 2 hr | 95% |
| Evaluation | 5 min | 90% |
| **Total** | **~4 hrs** | |

---

## 🎓 What's Happening?

1. **Data Loading**: Streaming prevents memory issues
2. **LoRA Training**: Only trains 1% of parameters
3. **Auto-save**: Checkpoints every 500 steps
4. **Evaluation**: Tests on 300 held-out images
5. **Plots**: Shows prediction accuracy

---

## 📁 What You'll Get

```
trained_models/
├── florence2_final/          # 230M param model
├── qwen2_5vl_final/          # 2B param model
├── evaluation_results.json   # MAE, RMSE, R²
├── Florence-2_evaluation.png # Prediction plot
└── Qwen2.5-VL_evaluation.png # Prediction plot
```

---

## 🚀 Advanced: All-in-One Runner

For fully automated pipeline:

```bash
# Make executable
chmod +x run_pipeline.py

# Run everything
python run_pipeline.py

# Or specific parts
python run_pipeline.py --train-only
python run_pipeline.py --demo-only
python run_pipeline.py --share
```

---

## 💡 Pro Tips

1. **Use `screen` or `tmux`** for long sessions
2. **Monitor GPU temp**: `nvidia-smi dmon`
3. **Save logs**: `python train_vision_models.py 2>&1 | tee log.txt`
4. **Background mode**: `nohup python train_vision_models.py &`

---

## 🎉 Success Checklist

- [ ] Environment setup complete
- [ ] HuggingFace logged in
- [ ] Dataset configured
- [ ] GPU detected (48GB)
- [ ] Training started
- [ ] TensorBoard running
- [ ] Models saved
- [ ] Evaluation complete
- [ ] Demo working

---

## 📞 Need Help?

1. Check `README.md` for detailed info
2. Check `COMMANDS.md` for all commands
3. Check error messages in terminal
4. Verify GPU: `nvidia-smi`

---

## ⏱️ Time Estimates (A6000)

| Dataset Size | Florence-2 | Qwen2.5-VL | Total |
|--------------|------------|------------|-------|
| 500 images | 30 min | 45 min | 1.5 hr |
| 1000 images | 1 hr | 1.5 hr | 3 hr |
| 2000 images | 2 hr | 3 hr | 5.5 hr |
| 5000 images | 5 hr | 7 hr | 13 hr |

---

## 🔑 Key Files Explained

| File | Purpose |
|------|---------|
| `train_vision_models.py` | Main training script |
| `gradio_demo.py` | Web demo interface |
| `requirements.txt` | Python dependencies |
| `setup.sh` | Automated setup |
| `prepare_dataset.py` | Dataset preparation |
| `run_pipeline.py` | All-in-one runner |
| `README.md` | Full documentation |
| `COMMANDS.md` | Command reference |

---

**Happy Training! 🚀**
