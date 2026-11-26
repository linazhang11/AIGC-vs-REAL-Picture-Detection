"""
Main entry point for AIGC Image Detection System.
"""
import argparse
import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from torchvision import transforms

from configs.config import Config
from data.dataset import load_data, get_class_names, get_transforms, load_data_auto
from models.resnet import create_resnet, get_model_info
from losses.focal_loss import get_loss_function
from training.trainer import Trainer, get_optimizer
from evaluation.evaluator import Evaluator
from visualization.visualizer import Visualizer
from utils.utils import set_seed, save_model, load_model


def train_mode(config: Config):
    """
    Run training mode.
    
    Args:
        config: Configuration object.
    """
    print("\n" + "=" * 60)
    print("AIGC IMAGE DETECTION SYSTEM - TRAINING MODE")
    print("=" * 60)
    
    # Set random seed
    set_seed(config.seed)
        
    # Print configuration
    print(f"\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
        
    # Load data (auto-select best method)
    print("\nLoading dataset...")
    train_loader, test_loader, train_gpu_transforms, test_gpu_transforms = load_data_auto(
        config.data_dir,
        batch_size=config.batch_size,
        use_gpu_transforms=config.use_gpu_transforms
    )
    class_names = get_class_names(config.data_dir)
    print(f"Class names: {class_names}")
    
    # Create model
    print(f"\nCreating model: {config.model_name}")
    model = create_resnet(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=config.pretrained
    ).to(config.device)
    
    # Print model info
    model_info = get_model_info(model)
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    # Create loss function
    criterion = get_loss_function(
        config.loss_function,
        config.focal_alpha,
        config.focal_gamma
    )
    print(f"\nLoss function: {config.loss_function}")
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        config.optimizer,
        config.learning_rate,
        config.momentum,
        config.weight_decay
    )
    print(f"Optimizer: {config.optimizer}")
    
    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        gpu_transforms=train_gpu_transforms
    )
        
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config.epochs,
        save_best_model=True,
        model_save_path=config.model_save_path,
        model_name=f"best_model_{config.model_name}.pth",
        test_gpu_transforms=test_gpu_transforms
    )
    
    # Save final model
    save_model(
        model=model,
        save_path=config.model_save_path,
        filename=f"final_model_{config.model_name}.pth",
        optimizer=optimizer,
        epoch=config.epochs
    )
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = os.path.join(config.model_save_path, f"best_model_{config.model_name}.pth")
    model, _, _, _ = load_model(model, best_model_path, config.device)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = Evaluator(model, config.device)
    evaluator.print_evaluation_results(test_loader, class_names)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = Visualizer(config.plot_save_path)
    
    # Training curves
    visualizer.plot_training_curves(
        train_losses=history["train_losses"],
        train_accuracies=history["train_accuracies"],
        test_accuracies=history["test_accuracies"],
        filename=f"training_curves_{config.model_name}.png",
        show=False
    )
    
    # Confusion matrix
    y_pred, y_true = evaluator.get_predictions(test_loader)
    visualizer.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        filename=f"confusion_matrix_{config.model_name}.png",
        show=False
    )
    
    # Sample predictions
    samples = evaluator.get_sample_predictions(test_loader, config.num_prediction_samples)
    visualizer.plot_predictions(
        samples=samples,
        class_names=class_names,
        filename=f"predictions_{config.model_name}.png",
        show=False
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Models saved to: {config.model_save_path}")
    print(f"Plots saved to: {config.plot_save_path}")


def evaluate_mode(config: Config, model_path: str):
    """
    Run evaluation mode.
    
    Args:
        config: Configuration object.
        model_path: Path to the saved model.
    """
    print("\n" + "=" * 60)
    print("AIGC IMAGE DETECTION SYSTEM - EVALUATION MODE")
    print("=" * 60)
    
    # Load data
    print("\nLoading test dataset...")
    _, test_loader = load_data(
        config.data_dir,
        batch_size=config.batch_size
    )
    class_names = get_class_names(config.data_dir)
    
    # Create model
    print(f"\nCreating model: {config.model_name}")
    model = create_resnet(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained=False
    ).to(config.device)
    
    # Load weights
    print(f"\nLoading model from: {model_path}")
    model, _, _, _ = load_model(model, model_path, config.device)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluator = Evaluator(model, config.device)
    evaluator.print_evaluation_results(test_loader, class_names)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualizer = Visualizer(config.plot_save_path)
    
    # Confusion matrix
    y_pred, y_true = evaluator.get_predictions(test_loader)
    visualizer.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        filename="eval_confusion_matrix.png",
        show=False
    )
    
    # Sample predictions
    samples = evaluator.get_sample_predictions(test_loader, config.num_prediction_samples)
    visualizer.plot_predictions(
        samples=samples,
        class_names=class_names,
        filename="eval_predictions.png",
        show=False
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED!")
    print("=" * 60)


def experiment_mode(config: Config):
    """
    Run experiment mode.
    
    Args:
        config: Configuration object.
    """
    print("\n" + "=" * 60)
    print("AIGC IMAGE DETECTION SYSTEM - EXPERIMENT MODE")
    print("=" * 60)
    
    from experiments.run_experiments import run_all_experiments
    
    results = run_all_experiments(config, skip_if_exists=True)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)
    print(f"Results saved to: {config.plot_save_path}")


def predict_mode(
    input_dir: str,
    model_path: str,
    output_dir: str = "./outputs/predictions",
    model_name: str = "resnet50",
    save_results: bool = True,
    verbose: bool = True
):
    """
    Run prediction mode - detect AIGC images in a folder.
    
    Args:
        input_dir: Directory containing images to detect.
        model_path: Path to the trained model.
        output_dir: Directory to save prediction results.
        model_name: Model architecture name.
        save_results: Whether to save results to file.
        verbose: Whether to print detailed output.
    """
    print("\n" + "=" * 60)
    print("AIGC IMAGE DETECTION SYSTEM - PREDICTION MODE")
    print("=" * 60)
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return None
    
    # Collect all images
    image_files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            image_files.append(f)
    
    if len(image_files) == 0:
        print(f"Error: No supported images found in {input_dir}")
        print(f"Supported formats: {supported_extensions}")
        return None
    
    print(f"\nFound {len(image_files)} images to detect")
    print(f"Input directory: {input_dir}")
    print(f"Model: {model_path}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_resnet(
        model_name=model_name,
        num_classes=2,
        pretrained=False
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    model, _, _, _ = load_model(model, model_path, device)
    model.eval()
    print("Model loaded successfully!")
    
    # Class names based on ImageFolder alphabetical ordering
    # fake folder comes first alphabetically, so fake=0, real=1
    class_names = ['fake', 'real']
    
    # Transform for inference
    transform = get_transforms(train=False)
    
    # Predictions
    results = []
    
    print("\n" + "-" * 60)
    print("DETECTION RESULTS")
    print("-" * 60)
    print(f"{'Filename':<40} {'Result':<10} {'Confidence':<10}")
    print("-" * 60)
    
    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                pred_class = class_names[predicted.item()]
                conf_value = confidence.item() * 100
                
                # Determine if AIGC
                is_aigc = predicted.item() == 0  # fake=0 (AIGC), real=1
                
                # Store result
                result = {
                    "filename": img_file,
                    "is_aigc": is_aigc,
                    "prediction": pred_class,
                    "confidence": round(conf_value, 2),
                    "prob_fake": round(probabilities[0][0].item() * 100, 2),
                    "prob_real": round(probabilities[0][1].item() * 100, 2)
                }
                results.append(result)
                
                # Print result
                if verbose:
                    status = "AIGC" if is_aigc else "REAL"
                    print(f"{img_file[:40]:<40} {status:<10} {conf_value:.1f}%")
                
            except Exception as e:
                print(f"{img_file:<40} ERROR: {str(e)[:30]}")
                results.append({
                    "filename": img_file,
                    "error": str(e)
                })
    
    print("-" * 60)
    
    # Summary
    total = len(results)
    aigc_count = sum(1 for r in results if r.get("is_aigc", False))
    real_count = total - aigc_count
    
    print(f"\nSUMMARY:")
    print(f"  Total images: {total}")
    print(f"  Detected as AIGC: {aigc_count}")
    print(f"  Detected as Real: {real_count}")
    
    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "input_dir": input_dir,
                "model_path": model_path,
                "total_images": total,
                "aigc_count": aigc_count,
                "real_count": real_count,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
        
        # Save TXT report
        txt_path = os.path.join(output_dir, f"report_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("AIGC Image Detection Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input: {input_dir}\n")
            f.write(f"Model: {model_path}\n")
            f.write("\n" + "-" * 60 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Filename':<40} {'Result':<10} {'Confidence':<10}\n")
            f.write("-" * 60 + "\n")
            for r in results:
                if "error" not in r:
                    status = "AIGC" if r["is_aigc"] else "REAL"
                    f.write(f"{r['filename'][:40]:<40} {status:<10} {r['confidence']:.1f}%\n")
            f.write("-" * 60 + "\n")
            f.write(f"\nSUMMARY:\n")
            f.write(f"  Total: {total}\n")
            f.write(f"  AIGC: {aigc_count}\n")
            f.write(f"  Real: {real_count}\n")
        print(f"Report saved to: {txt_path}")
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETED!")
    print("=" * 60)
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AIGC Image Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["train", "evaluate", "experiment", "predict"],
        help="Running mode: train, evaluate, experiment, or predict (default: predict)"
    )
        
    # Data and output paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Path to the dataset directory (default: ./dataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to save outputs (default: ./outputs)"
    )
        
    # Prediction mode specific
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input_images",
        help="Directory containing images to detect (for predict mode)"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50"],
        help="Model architecture (default: resnet50)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model for evaluation mode"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20)"
    )
    
    # Loss function
    parser.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "focal_loss"],
        help="Loss function (default: cross_entropy)"
    )
    
    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer (default: adam)"
    )
    
    # Other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_name=args.model,
        loss_function=args.loss,
        optimizer=args.optimizer,
        seed=args.seed,
        early_stopping_patience=args.patience
    )
    
    # Run selected mode
    if args.mode == "train":
        train_mode(config)
    elif args.mode == "evaluate":
        if args.model_path is None:
            print("Error: --model_path is required for evaluation mode")
            sys.exit(1)
        evaluate_mode(config, args.model_path)
    elif args.mode == "experiment":
        experiment_mode(config)
    elif args.mode == "predict":
        # Find default model if not specified
        model_path = args.model_path
        if model_path is None:
            # Look for default model
            default_models = [
                os.path.join(config.model_save_path, "baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth"),
                os.path.join(config.model_save_path, "best_model_resnet50.pth"),
                os.path.join(config.model_save_path, "final_model_resnet50.pth"),
                "./outputs/models/best_model_resnet50.pth",
            ]
            for mp in default_models:
                if os.path.exists(mp):
                    model_path = mp
                    break
            
        if model_path is None or not os.path.exists(model_path):
            print("Error: No model found. Please train a model first or specify --model_path")
            print("Usage: python main.py --mode predict --input_dir ./input_images --model_path ./outputs/models/best_model_resnet50.pth")
            sys.exit(1)
            
        predict_mode(
            input_dir=args.input_dir,
            model_path=model_path,
            output_dir=os.path.join(args.output_dir, "predictions"),
            model_name=args.model
        )


if __name__ == "__main__":
    main()
