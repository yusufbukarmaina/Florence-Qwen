"""
Dataset Preparation Script for Beaker Volume Dataset
Prepares images and annotations for HuggingFace upload
"""

import os
import json
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from typing import List, Dict
import argparse

# ============================================================================
# DATASET PREPARATION
# ============================================================================

class DatasetPreparator:
    """Prepare beaker dataset for HuggingFace upload"""
    
    def __init__(self, data_dir: str, output_dir: str = "./prepared_dataset"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_from_images_and_labels(
        self,
        image_dir: str,
        labels_file: str,
        format: str = "json"
    ):
        """
        Prepare dataset from image directory and labels file
        
        Args:
            image_dir: Directory containing beaker images
            labels_file: JSON/CSV file with image_name -> volume mapping
            format: 'json' or 'csv'
        """
        print(f"📂 Loading images from: {image_dir}")
        print(f"📄 Loading labels from: {labels_file}")
        
        # Load labels
        if format == "json":
            with open(labels_file, 'r') as f:
                labels = json.load(f)
        elif format == "csv":
            import pandas as pd
            df = pd.read_csv(labels_file)
            labels = dict(zip(df['image_name'], df['volume']))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Prepare dataset
        data = []
        
        for image_name, volume in labels.items():
            image_path = Path(image_dir) / image_name
            
            if not image_path.exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                continue
            
            # Load and verify image
            try:
                img = Image.open(image_path).convert('RGB')
                
                data.append({
                    'image': img,
                    'volume': f"{volume} mL",
                    'answer': f"The volume of liquid in the beaker is {volume} mL.",
                    'background': self._detect_background(image_name),
                    'image_name': image_name
                })
                
            except Exception as e:
                print(f"❌ Error loading {image_path}: {e}")
                continue
        
        print(f"✅ Prepared {len(data)} examples")
        
        return data
    
    def prepare_from_directory_structure(self, root_dir: str):
        """
        Prepare dataset from directory structure:
        
        root_dir/
        ├── normal_background/
        │   ├── beaker_250ml_001.jpg
        │   └── beaker_125ml_002.jpg
        └── cluttered_background/
            ├── beaker_500ml_001.jpg
            └── beaker_375ml_002.jpg
        
        Volume is extracted from filename (e.g., "beaker_250ml_001.jpg" -> 250)
        """
        print(f"📂 Scanning directory structure: {root_dir}")
        
        data = []
        root_path = Path(root_dir)
        
        # Scan all subdirectories
        for background_type in ['normal_background', 'cluttered_background', 'normal', 'cluttered']:
            background_dir = root_path / background_type
            
            if not background_dir.exists():
                continue
            
            print(f"  Processing: {background_type}/")
            
            for image_path in background_dir.glob("*.jpg"):
                try:
                    # Extract volume from filename
                    volume = self._extract_volume_from_filename(image_path.name)
                    
                    if volume is None:
                        print(f"⚠️  Could not extract volume from: {image_path.name}")
                        continue
                    
                    # Load image
                    img = Image.open(image_path).convert('RGB')
                    
                    data.append({
                        'image': img,
                        'volume': f"{volume} mL",
                        'answer': f"The volume of liquid in the beaker is {volume} mL.",
                        'background': background_type.replace('_background', ''),
                        'image_name': image_path.name
                    })
                    
                except Exception as e:
                    print(f"❌ Error loading {image_path}: {e}")
                    continue
        
        print(f"✅ Prepared {len(data)} examples")
        
        return data
    
    def _extract_volume_from_filename(self, filename: str) -> float:
        """Extract volume from filename like 'beaker_250ml_001.jpg'"""
        import re
        
        # Try different patterns
        patterns = [
            r'(\d+\.?\d*)ml',
            r'(\d+\.?\d*)mL',
            r'(\d+\.?\d*)_ml',
            r'vol_?(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _detect_background(self, filename: str) -> str:
        """Detect background type from filename"""
        filename_lower = filename.lower()
        
        if 'clutter' in filename_lower:
            return 'cluttered'
        elif 'normal' in filename_lower:
            return 'normal'
        else:
            return 'unknown'
    
    def create_huggingface_dataset(self, data: List[Dict]) -> Dataset:
        """Convert prepared data to HuggingFace Dataset"""
        print("🤗 Creating HuggingFace Dataset...")
        
        # Define features
        features = Features({
            'image': HFImage(),
            'volume': Value('string'),
            'answer': Value('string'),
            'background': Value('string'),
            'image_name': Value('string')
        })
        
        # Create dataset
        dataset = Dataset.from_dict(
            {
                'image': [d['image'] for d in data],
                'volume': [d['volume'] for d in data],
                'answer': [d['answer'] for d in data],
                'background': [d['background'] for d in data],
                'image_name': [d['image_name'] for d in data]
            },
            features=features
        )
        
        print(f"✅ Created dataset with {len(dataset)} examples")
        
        return dataset
    
    def save_to_disk(self, dataset: Dataset, output_path: str = None):
        """Save dataset to disk"""
        if output_path is None:
            output_path = self.output_dir / "dataset"
        
        print(f"💾 Saving dataset to: {output_path}")
        dataset.save_to_disk(str(output_path))
        print("✅ Dataset saved!")
    
    def push_to_hub(
        self,
        dataset: Dataset,
        repo_name: str,
        private: bool = False
    ):
        """Upload dataset to HuggingFace Hub"""
        print(f"📤 Uploading to HuggingFace Hub: {repo_name}")
        
        dataset.push_to_hub(
            repo_name,
            private=private
        )
        
        print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")


# ============================================================================
# EXAMPLE USAGE FUNCTIONS
# ============================================================================

def example_1_from_directory():
    """
    Example 1: Prepare from directory structure
    
    Your images should be organized like:
    data/
    ├── normal_background/
    │   ├── beaker_250ml_001.jpg
    │   └── beaker_125ml_002.jpg
    └── cluttered_background/
        ├── beaker_500ml_001.jpg
        └── beaker_375ml_002.jpg
    """
    preparator = DatasetPreparator(data_dir="./data")
    
    # Prepare data
    data = preparator.prepare_from_directory_structure("./data")
    
    # Create HuggingFace dataset
    dataset = preparator.create_huggingface_dataset(data)
    
    # Save locally (optional)
    preparator.save_to_disk(dataset)
    
    # Upload to HuggingFace
    preparator.push_to_hub(
        dataset,
        repo_name="your-username/beaker-volume-dataset",
        private=False
    )


def example_2_from_json():
    """
    Example 2: Prepare from images + JSON labels
    
    labels.json format:
    {
        "beaker_001.jpg": 250.0,
        "beaker_002.jpg": 125.5,
        "beaker_003.jpg": 500.0
    }
    """
    preparator = DatasetPreparator(data_dir="./data")
    
    # Prepare data
    data = preparator.prepare_from_images_and_labels(
        image_dir="./data/images",
        labels_file="./data/labels.json",
        format="json"
    )
    
    # Create and upload
    dataset = preparator.create_huggingface_dataset(data)
    preparator.push_to_hub(
        dataset,
        repo_name="your-username/beaker-volume-dataset",
        private=False
    )


def example_3_from_csv():
    """
    Example 3: Prepare from images + CSV labels
    
    labels.csv format:
    image_name,volume
    beaker_001.jpg,250.0
    beaker_002.jpg,125.5
    beaker_003.jpg,500.0
    """
    preparator = DatasetPreparator(data_dir="./data")
    
    # Prepare data
    data = preparator.prepare_from_images_and_labels(
        image_dir="./data/images",
        labels_file="./data/labels.csv",
        format="csv"
    )
    
    # Create and upload
    dataset = preparator.create_huggingface_dataset(data)
    preparator.push_to_hub(
        dataset,
        repo_name="your-username/beaker-volume-dataset",
        private=False
    )


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare beaker volume dataset for HuggingFace"
    )
    
    parser.add_argument(
        "--mode",
        choices=["directory", "json", "csv"],
        required=True,
        help="Dataset preparation mode"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing images"
    )
    
    parser.add_argument(
        "--labels-file",
        type=str,
        help="Path to labels file (for json/csv modes)"
    )
    
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="HuggingFace repository name (e.g., username/dataset-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private"
    )
    
    parser.add_argument(
        "--save-local",
        action="store_true",
        help="Save dataset locally before uploading"
    )
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = DatasetPreparator(data_dir=args.data_dir)
    
    # Prepare data based on mode
    if args.mode == "directory":
        data = preparator.prepare_from_directory_structure(args.data_dir)
    
    elif args.mode in ["json", "csv"]:
        if not args.labels_file:
            raise ValueError(f"--labels-file required for {args.mode} mode")
        
        data = preparator.prepare_from_images_and_labels(
            image_dir=args.data_dir,
            labels_file=args.labels_file,
            format=args.mode
        )
    
    # Create HuggingFace dataset
    dataset = preparator.create_huggingface_dataset(data)
    
    # Show dataset info
    print("\n" + "="*80)
    print("📊 Dataset Info:")
    print("="*80)
    print(dataset)
    print(f"\nFeatures: {dataset.features}")
    print(f"Number of examples: {len(dataset)}")
    
    # Save locally if requested
    if args.save_local:
        preparator.save_to_disk(dataset)
    
    # Upload to HuggingFace
    print("\n" + "="*80)
    print("📤 Uploading to HuggingFace...")
    print("="*80)
    
    preparator.push_to_hub(
        dataset,
        repo_name=args.repo_name,
        private=args.private
    )
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\nYour dataset is available at:")
    print(f"https://huggingface.co/datasets/{args.repo_name}")
    print(f"\nUse it in training with:")
    print(f'HF_DATASET_NAME = "{args.repo_name}"')


if __name__ == "__main__":
    # If running without arguments, show examples
    import sys
    
    if len(sys.argv) == 1:
        print("="*80)
        print("🧪 Beaker Volume Dataset Preparation Tool")
        print("="*80)
        print("\nUsage examples:\n")
        
        print("1. From directory structure:")
        print("   python prepare_dataset.py \\")
        print("     --mode directory \\")
        print("     --data-dir ./my_beaker_images \\")
        print("     --repo-name username/beaker-dataset")
        
        print("\n2. From JSON labels:")
        print("   python prepare_dataset.py \\")
        print("     --mode json \\")
        print("     --data-dir ./images \\")
        print("     --labels-file ./labels.json \\")
        print("     --repo-name username/beaker-dataset")
        
        print("\n3. From CSV labels:")
        print("   python prepare_dataset.py \\")
        print("     --mode csv \\")
        print("     --data-dir ./images \\")
        print("     --labels-file ./labels.csv \\")
        print("     --repo-name username/beaker-dataset \\")
        print("     --private")
        
        print("\nFor more help:")
        print("   python prepare_dataset.py --help")
        
        sys.exit(0)
    
    main()
