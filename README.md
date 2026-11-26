# AIGC Image Detection System

AI Generated Image Detection System - Detect whether an image is real or AI-generated.

## Features

- **Binary Classification**: Detects if an image is real (0) or AI-generated (1)
- **Confidence Score**: Provides probability confidence for each prediction
- **Auto Download**: Automatically downloads dataset from Kaggle if not found
- **Multiple Modes**: Training, Evaluation, Prediction, and Experiment modes

## Quick Start

### 1. Installation

#### CPU Version
```bash
cd aigc_detection
pip install -r requirements.txt
```

#### GPU Version (Recommended for RTX 4070 / CUDA)

**Step 1: Uninstall CPU version (if installed)**
```bash
pip uninstall torch torchvision torchaudio -y
```

**Step 2: Install GPU version**
```bash
# CUDA 11.8 (Recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 3: Verify GPU availability**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Step 4: Install other dependencies**
```bash
pip install numpy matplotlib scikit-learn tqdm seaborn kagglehub
```

### 2. Train the Model (First Time)

```bash
# Train the baseline model
python main.py --mode train
```

The system will automatically:
- Download the dataset from Kaggle (if not found)
- Train the ResNet50 model with pretrained ImageNet weights
- Save the model to `./outputs/models/`

### 3. Detect Images

Put your images in the `input_images` folder, then run:

```bash
python predict.py
```

Or specify a custom folder:

```bash
python predict.py --input ./my_images
```

## Usage Examples

### Prediction Mode (Recommended for Users)

```bash
# Use default input_images folder
python predict.py

# Specify custom input folder
python predict.py --input ./my_images

# Specify model path (new naming convention)
python predict.py --input ./my_images --model ./outputs/models/baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth
```

### Training Mode

```bash
# Train with default settings (ResNet50)
python main.py --mode train

# Train with custom settings
python main.py --mode train --epochs 30 --batch_size 64 --lr 0.001
```

### Evaluation Mode

```bash
# Evaluate a trained model
python main.py --mode evaluate --model_path ./outputs/models/baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth
```

> **Note**: The system uses ResNet50 architecture only.

### Experiment Mode

Run comparison experiments with different hyperparameters:

```bash
# Run all 6 comparison experiments
python main.py --mode experiment
```

**Experiment Schedule:**

| No. | Name | Batch | Epoch | LR | Optimizer | Loss |
|-----|------|-------|-------|-----|-----------|------|
| 1 | baseline | 64 | 30 | 0.001 | Adam | CrossEntropy |
| 2 | lr_compare | 64 | 30 | 0.0001 | Adam | CrossEntropy |
| 3 | batch_compare | 32 | 30 | 0.001 | Adam | CrossEntropy |
| 4 | loss_compare | 64 | 30 | 0.001 | Adam | FocalLoss |
| 5 | optimizer_compare | 64 | 30 | 0.001 | SGD | CrossEntropy |
| 6 | epoch_compare | 64 | 20 | 0.001 | Adam | CrossEntropy |

**Model Naming Convention:**

Models are saved with descriptive names including all training parameters:
```
{name}_B{batch}_E{epoch}_L{lr}_O{optimizer}_L{loss}.pth
```

Example: `baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth`

**Skip Existing Models:**

The experiment mode automatically skips models that already exist, avoiding redundant training.

**Output Files:**

Each experiment generates the following files:

**Models:**
```
outputs/models/
├── {name}_B{batch}_E{epoch}_L{lr}_O{optimizer}_L{loss}.pth        # Best model
└── {name}_B{batch}_E{epoch}_L{lr}_O{optimizer}_L{loss}_final.pth   # Final model
```

**Plots:**
```
outputs/plots/
├── {name}_B{batch}_E{epoch}_L{lr}_O{optimizer}_L{loss}_training_curves.png     # Training curves
├── {name}_B{batch}_E{epoch}_L{lr}_O{optimizer}_L{loss}_confusion_matrix.png    # Confusion matrix
├── all_experiments_comparison.png                                               # Final accuracy comparison
└── all_experiments_accuracy_comparison.png                                      # Accuracy curves comparison
```

Example:
- `baseline_B64_E30_L0.001_OAdam_LCrossEntropy_training_curves.png`
- `baseline_B64_E30_L0.001_OAdam_LCrossEntropy_confusion_matrix.png`

## Output Format

### Prediction Results

The system outputs detection results in two formats:

**Console Output:**
```
------------------------------------------------------------
DETECTION RESULTS
------------------------------------------------------------
Filename                                Result     Confidence
------------------------------------------------------------
image1.jpg                              AIGC       95.3%
image2.png                              REAL       87.2%
image3.jpg                              AIGC       99.1%
------------------------------------------------------------

SUMMARY:
  Total images: 3
  Detected as AIGC: 2
  Detected as Real: 1
```

**JSON Output (`results_TIMESTAMP.json`):**
```json
{
  "timestamp": "20240101_120000",
  "input_dir": "./input_images",
  "total_images": 3,
  "aigc_count": 2,
  "real_count": 1,
  "results": [
    {
      "filename": "image1.jpg",
      "is_aigc": true,
      "prediction": "fake",
      "confidence": 95.3,
      "prob_real": 4.7,
      "prob_fake": 95.3
    }
  ]
}
```

## Project Structure

```
aigc_detection/
├── configs/           # Configuration management
├── data/              # Data loading and preprocessing
├── models/            # Model definitions (ResNet50)
├── losses/            # Loss functions (CrossEntropy, Focal Loss)
├── training/          # Training loop and optimizer
├── evaluation/        # Evaluation metrics
├── visualization/     # Plotting and visualization
├── utils/             # Utility functions
├── experiments/       # Experiment scripts (6 comparison experiments)
├── main.py            # Main entry point
├── predict.py         # Simple prediction script
├── requirements.txt   # Dependencies
└── input_images/      # Default folder for images to detect
```

## Command Line Arguments

### Main Script (main.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | predict | Running mode: train, evaluate, experiment, predict |
| `--data_dir` | ./dataset | Dataset directory |
| `--input_dir` | ./input_images | Input images for prediction |
| `--output_dir` | ./outputs | Output directory |
| `--model` | resnet50 | Model architecture (resnet50 only) |
| `--model_path` | auto | Path to saved model |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--loss` | cross_entropy | Loss: cross_entropy or focal_loss |
| `--optimizer` | adam | Optimizer: adam or sgd |
| `--patience` | 5 | Early stopping patience |

### Prediction Script (predict.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input, -i` | ./input_images | Input images folder |
| `--model, -m` | auto | Model path |
| `--output, -o` | ./outputs/predictions | Results output folder |
| `--quiet, -q` | False | Suppress detailed output |

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)
- TIFF (.tiff)

## Model Performance

The model achieves high accuracy on the AI-generated vs Real image classification task:

- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Training Dataset**: [Kaggle - AI Generated Images vs Real Images](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
- **Expected Accuracy**: ~95%+ on test set

## Requirements

- Python 3.8+
- PyTorch 2.0+ (GPU version recommended)
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- seaborn
- kagglehub
- PIL / Pillow

## Hardware Recommendations

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | - | RTX 4070 / CUDA |
| Storage | 50 GB | 100+ GB (for dataset) |

## Training Time Estimate

| Device | Time per Epoch | Total (20 epochs) |
|--------|----------------|-------------------|
| CPU | ~45 min | ~15 hours |
| RTX 4070 | ~2 min | ~40 min |

## License

This project is for educational purposes.
