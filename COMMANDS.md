# 🚀 Quick Command Reference

## Initial Setup (Run Once)

```bash
# 1. Make setup script executable
chmod +x setup.sh

# 2. Run setup
./setup.sh

# 3. Login to HuggingFace (when prompted)
# Get token from: https://huggingface.co/settings/tokens

# 4. Edit configuration
nano train_vision_models.py
# Update lines 25 and 49 with your dataset/repo names
```

## Training Commands

```bash
# Start training (main process)
python train_vision_models.py

# Monitor with TensorBoard (in new terminal)
tensorboard --logdir=./trained_models --port=6006

# Check GPU usage (in new terminal)
watch -n 1 nvidia-smi
```

## After Training

```bash
# Launch Gradio demo locally
python gradio_demo.py

# Launch with public sharing link
python gradio_demo.py --share

# Launch on specific port
python gradio_demo.py --port 8080 --share

# Specify custom model paths
python gradio_demo.py \
  --florence-path ./trained_models/florence2_final \
  --qwen-path ./trained_models/qwen2_5vl_final \
  --share
```

## Useful Monitoring Commands

```bash
# Check GPU memory usage
nvidia-smi

# Monitor GPU continuously
watch -n 1 nvidia-smi

# Check disk space
df -h

# Monitor training logs
tail -f ./trained_models/*/trainer_state.json

# View TensorBoard logs
ls -lah ./trained_models/*/runs/
```

## File Management

```bash
# Check model sizes
du -sh ./trained_models/*

# List all saved checkpoints
ls -lah ./trained_models/florence2/
ls -lah ./trained_models/qwen2_5vl/

# View evaluation results
cat ./trained_models/evaluation_results.json | python -m json.tool

# View evaluation plots
ls -lah ./trained_models/*.png
```

## Troubleshooting Commands

```bash
# Check Python packages
pip list | grep -E "torch|transformers|peft|datasets"

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Test model loading
python -c "from transformers import AutoProcessor; print('OK')"

# Clear CUDA cache (if OOM)
python -c "import torch; torch.cuda.empty_cache()"
```

## HuggingFace Commands

```bash
# Login to HuggingFace
huggingface-cli login

# Check login status
huggingface-cli whoami

# Upload model manually
huggingface-cli upload your-username/model-name ./trained_models/florence2_final

# Download your dataset
huggingface-cli download your-username/dataset-name
```

## Advanced: Resume Training

If training is interrupted:

```bash
# Training will automatically resume from last checkpoint
python train_vision_models.py

# To start fresh, remove checkpoints
rm -rf ./trained_models/florence2/checkpoint-*
rm -rf ./trained_models/qwen2_5vl/checkpoint-*
```

## Advanced: Custom Training

```bash
# Train only Florence-2 (edit train_vision_models.py)
# Comment out the Qwen training section (lines ~300-320)

# Train only Qwen2.5-VL (edit train_vision_models.py)
# Comment out the Florence training section (lines ~280-300)

# Change number of epochs
# Edit line ~42: NUM_EPOCHS = 15

# Change batch size
# Edit line ~38: BATCH_SIZE = 2
```

## Port Forwarding (JarvisLab)

If you need to access services from your local machine:

```bash
# On your local machine, SSH tunnel to JarvisLab
ssh -L 6006:localhost:6006 -L 7860:localhost:7860 jarvislab

# Then access:
# TensorBoard: http://localhost:6006
# Gradio: http://localhost:7860
```

## Clean Up

```bash
# Remove checkpoints to save space (keep final models)
rm -rf ./trained_models/*/checkpoint-*

# Remove TensorBoard logs
rm -rf ./trained_models/*/runs/

# Remove cache files
rm -rf ~/.cache/huggingface/hub/

# Full clean (WARNING: removes all trained models)
rm -rf ./trained_models/
```

## Quick Tests

```bash
# Test dataset loading
python -c "
from datasets import load_dataset
ds = load_dataset('your-username/beaker-dataset', split='train', streaming=True)
print(next(iter(ds)))
"

# Test Florence-2 loading
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True
)
print('Florence-2 loaded successfully')
"

# Test Qwen2.5-VL loading
python -c "
from transformers import Qwen2VLForConditionalGeneration
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    trust_remote_code=True
)
print('Qwen2.5-VL loaded successfully')
"
```

## Time Estimates (A6000 GPU)

- **Setup**: 10-15 minutes
- **Florence-2 Training**: 1-2 hours (10 epochs, 1400 samples)
- **Qwen2.5-VL Training**: 1.5-2.5 hours (10 epochs, 1400 samples)
- **Evaluation**: 5-10 minutes (300 test samples)
- **Total**: ~3-5 hours

## Memory Usage

- **Florence-2**: ~2 GB base + ~12 GB training = 14 GB
- **Qwen2.5-VL**: ~8 GB base + ~12 GB training = 20 GB
- **Peak Total**: ~25 GB (safe for 48 GB A6000)

## Emergency Stop

```bash
# Stop training gracefully (Ctrl+C)
# Model will save current checkpoint

# Force kill if frozen
pkill -9 python

# Check for zombie processes
ps aux | grep python
```

## Tips

1. **Always use `screen` or `tmux`** for long-running training:
   ```bash
   screen -S training
   python train_vision_models.py
   # Detach: Ctrl+A, D
   # Reattach: screen -r training
   ```

2. **Monitor GPU temperature**:
   ```bash
   nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
   ```

3. **Save terminal output**:
   ```bash
   python train_vision_models.py 2>&1 | tee training.log
   ```

4. **Background training**:
   ```bash
   nohup python train_vision_models.py > training.log 2>&1 &
   tail -f training.log
   ```
