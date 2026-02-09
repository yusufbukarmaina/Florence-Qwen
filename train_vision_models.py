"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
Optimized for JarvisLab A6000 GPU with streaming dataset loading
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset settings
    HF_DATASET_NAME = "YOUR_USERNAME/beaker-volume-dataset"  # REPLACE WITH YOUR DATASET
    STREAMING = True  # Enable streaming to avoid OOM
    
    # Model settings
    FLORENCE_MODEL = "microsoft/Florence-2-base"  # Smaller version
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # 2B version (smallest)
    
    # Training split
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Training settings
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    MAX_LENGTH = 512
    
    # Output settings
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    # HuggingFace upload
    UPLOAD_TO_HF = False  # Set to True to upload after training
    HF_REPO_NAME = "YOUR_USERNAME/beaker-volume-model"  # REPLACE


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DatasetProcessor:
    """Process and split dataset with streaming support"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        """Load dataset with streaming and create splits"""
        print("📥 Loading dataset with streaming...")
        
        # Load full dataset in streaming mode
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        )
        
        # For streaming datasets, we need to shuffle and split differently
        if self.config.STREAMING:
            # Shuffle the dataset
            dataset = dataset.shuffle(seed=42, buffer_size=1000)
            
            # Since we can't directly split streaming datasets by percentage,
            # we'll take first N examples for each split
            # We'll process in batches and route to appropriate splits
            print("📊 Creating train/val/test splits with streaming...")
            
            # We'll collect splits in memory in batches
            train_data = []
            val_data = []
            test_data = []
            
            total_processed = 0
            batch_size = 100
            
            for example in dataset:
                total_processed += 1
                
                # Determine which split this example belongs to
                rand_val = hash(str(total_processed)) % 100 / 100.0
                
                if rand_val < self.config.TRAIN_SPLIT:
                    train_data.append(example)
                elif rand_val < (self.config.TRAIN_SPLIT + self.config.VAL_SPLIT):
                    val_data.append(example)
                else:
                    test_data.append(example)
                
                # Print progress every 100 examples
                if total_processed % batch_size == 0:
                    print(f"Processed {total_processed} examples - "
                          f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
                
                # Stop after processing 2000 examples (adjust as needed)
                if total_processed >= 2000:
                    break
            
            print(f"\n✅ Dataset split complete:")
            print(f"   Train: {len(train_data)} examples")
            print(f"   Val: {len(val_data)} examples")
            print(f"   Test: {len(test_data)} examples")
            
            return train_data, val_data, test_data
        
        else:
            # Non-streaming mode (loads all data into memory)
            dataset = dataset.shuffle(seed=42)
            
            total_size = len(dataset)
            train_size = int(total_size * self.config.TRAIN_SPLIT)
            val_size = int(total_size * self.config.VAL_SPLIT)
            
            train_data = dataset.select(range(train_size))
            val_data = dataset.select(range(train_size, train_size + val_size))
            test_data = dataset.select(range(train_size + val_size, total_size))
            
            return list(train_data), list(val_data), list(test_data)
    
    def extract_volume_from_text(self, text: str) -> float:
        """Extract volume value from text answer"""
        # Look for patterns like "250 mL", "250mL", "250.5 mL", etc.
        patterns = [
            r'(\d+\.?\d*)\s*mL',
            r'(\d+\.?\d*)\s*ml',
            r'(\d+\.?\d*)\s*milliliters?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # If no pattern found, try to extract any number
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        
        return 0.0


# ============================================================================
# FLORENCE-2 TRAINING
# ============================================================================

class FlorenceTrainer:
    """Florence-2 model trainer with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        """Initialize Florence-2 model with LoRA"""
        print(f"\n🤖 Setting up Florence-2 model: {self.config.FLORENCE_MODEL}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.processor
    
    def preprocess_data(self, examples: List[Dict]) -> Dict:
        """Preprocess data for Florence-2"""
        images = []
        texts = []
        
        for example in examples:
            # Load image
            if isinstance(example['image'], str):
                image = Image.open(example['image']).convert('RGB')
            else:
                image = example['image'].convert('RGB')
            
            images.append(image)
            
            # Create prompt
            prompt = f"<VQA>What is the volume of liquid in the beaker?"
            answer = example.get('answer', example.get('volume', ''))
            
            text = f"{prompt}{answer}"
            texts.append(text)
        
        # Process with Florence processor
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_LENGTH
        )
        
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Train Florence-2 model"""
        print(f"\n🚀 Starting Florence-2 training...")
        
        # Setup model
        self.setup_model()
        
        # Create dataset class
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Load image
                if isinstance(example['image'], str):
                    image = Image.open(example['image']).convert('RGB')
                else:
                    image = example['image'].convert('RGB')
                
                # Create prompt and answer
                prompt = "<VQA>What is the volume of liquid in the beaker?"
                answer = example.get('answer', example.get('volume', ''))
                text = f"{prompt}{answer}"
                
                # Process
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                return inputs
        
        train_dataset = FlorenceDataset(train_data, self.processor, self.config)
        val_dataset = FlorenceDataset(val_data, self.processor, self.config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/florence2",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            report_to="tensorboard",
            save_total_limit=3,
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save final model
        final_path = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_path)
        self.processor.save_pretrained(final_path)
        
        print(f"✅ Florence-2 training complete! Model saved to {final_path}")
        
        return final_path


# ============================================================================
# QWEN2.5-VL TRAINING
# ============================================================================

class QwenTrainer:
    """Qwen2.5-VL model trainer with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        """Initialize Qwen2.5-VL model with LoRA"""
        print(f"\n🤖 Setting up Qwen2.5-VL model: {self.config.QWEN_MODEL}")
        
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True
        )
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.processor
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Train Qwen2.5-VL model"""
        print(f"\n🚀 Starting Qwen2.5-VL training...")
        
        # Setup model
        self.setup_model()
        
        # Create dataset class
        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Load image
                if isinstance(example['image'], str):
                    image = Image.open(example['image']).convert('RGB')
                else:
                    image = example['image'].convert('RGB')
                
                # Create conversation format
                question = "What is the volume of liquid in this beaker in mL?"
                answer = example.get('answer', example.get('volume', ''))
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer}
                        ]
                    }
                ]
                
                # Process
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                return inputs
        
        train_dataset = QwenDataset(train_data, self.processor, self.config)
        val_dataset = QwenDataset(val_data, self.processor, self.config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/qwen2_5vl",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            report_to="tensorboard",
            save_total_limit=3,
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save final model
        final_path = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_path)
        self.processor.save_pretrained(final_path)
        
        print(f"✅ Qwen2.5-VL training complete! Model saved to {final_path}")
        
        return final_path


# ============================================================================
# EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate models with MAE, RMSE, R2"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)
    
    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str):
        """Evaluate a model on test data"""
        print(f"\n📊 Evaluating {model_name}...")
        
        predictions = []
        ground_truth = []
        
        model.eval()
        
        with torch.no_grad():
            for example in test_data:
                # Load image
                if isinstance(example['image'], str):
                    image = Image.open(example['image']).convert('RGB')
                else:
                    image = example['image'].convert('RGB')
                
                # Get ground truth
                gt_text = example.get('answer', example.get('volume', ''))
                gt_volume = self.data_processor.extract_volume_from_text(gt_text)
                ground_truth.append(gt_volume)
                
                # Generate prediction
                if 'florence' in model_name.lower():
                    prompt = "<VQA>What is the volume of liquid in the beaker?"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                    
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3
                    )
                    
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                else:  # Qwen
                    question = "What is the volume of liquid in this beaker in mL?"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": question}
                            ]
                        }
                    ]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                    
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3
                    )
                    
                    generated_text = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                
                # Extract predicted volume
                pred_volume = self.data_processor.extract_volume_from_text(generated_text)
                predictions.append(pred_volume)
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        r2 = r2_score(ground_truth, predictions)
        
        print(f"\n📈 {model_name} Results:")
        print(f"   MAE:  {mae:.2f} mL")
        print(f"   RMSE: {rmse:.2f} mL")
        print(f"   R²:   {r2:.4f}")
        
        # Create visualization
        self.plot_predictions(ground_truth, predictions, model_name)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
    
    def plot_predictions(self, ground_truth, predictions, model_name):
        """Create prediction plots"""
        plt.figure(figsize=(10, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.plot([ground_truth.min(), ground_truth.max()], 
                 [ground_truth.min(), ground_truth.max()], 
                 'r--', lw=2)
        plt.xlabel('Ground Truth (mL)')
        plt.ylabel('Predictions (mL)')
        plt.title(f'{model_name} - Predictions vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = predictions - ground_truth
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (mL)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Error Distribution')
        plt.axvline(x=0, color='r', linestyle='--', lw=2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.config.OUTPUT_DIR}/{model_name.replace(' ', '_')}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Plot saved to: {plot_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL")
    print("="*80)
    
    # Initialize config
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and split dataset
    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()
    
    # Train Florence-2
    print("\n" + "="*80)
    print("FLORENCE-2 TRAINING")
    print("="*80)
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)
    
    # Train Qwen2.5-VL
    print("\n" + "="*80)
    print("QWEN2.5-VL TRAINING")
    print("="*80)
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)
    
    # Evaluate both models
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    evaluator = ModelEvaluator(config)
    
    # Evaluate Florence-2
    florence_results = evaluator.evaluate_model(
        florence_trainer.model,
        florence_trainer.processor,
        test_data,
        "Florence-2"
    )
    
    # Evaluate Qwen2.5-VL
    qwen_results = evaluator.evaluate_model(
        qwen_trainer.model,
        qwen_trainer.processor,
        test_data,
        "Qwen2.5-VL"
    )
    
    # Save results
    results = {
        'florence2': florence_results,
        'qwen2_5vl': qwen_results,
        'config': {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    results_path = f"{config.OUTPUT_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Upload to HuggingFace (optional)
    if config.UPLOAD_TO_HF:
        print("\n" + "="*80)
        print("UPLOADING TO HUGGINGFACE")
        print("="*80)
        
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Upload Florence-2
        print("📤 Uploading Florence-2...")
        api.upload_folder(
            folder_path=florence_path,
            repo_id=f"{config.HF_REPO_NAME}-florence2",
            repo_type="model"
        )
        
        # Upload Qwen2.5-VL
        print("📤 Uploading Qwen2.5-VL...")
        api.upload_folder(
            folder_path=qwen_path,
            repo_id=f"{config.HF_REPO_NAME}-qwen2-5vl",
            repo_type="model"
        )
        
        print("✅ Models uploaded to HuggingFace!")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to:")
    print(f"  Florence-2:   {florence_path}")
    print(f"  Qwen2.5-VL:   {qwen_path}")
    print(f"\nResults:       {results_path}")
    

if __name__ == "__main__":
    main()
