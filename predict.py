"""
Simple prediction script for AIGC Image Detection.

Usage:
    python predict.py                    # Use default input_images folder
    python predict.py --input my_images  # Specify custom folder
    python predict.py --input ./my_images --model ./outputs/models/baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from main import predict_mode


def main():
    parser = argparse.ArgumentParser(
        description="AIGC Image Detection - Simple Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py
    python predict.py --input ./my_images
    python predict.py --input ./my_images --model ./outputs/models/baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./input_images",
        help="Directory containing images to detect (default: ./input_images)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained model (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./outputs/predictions",
        help="Directory to save results (default: ./outputs/predictions)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Find model (prioritize new naming convention)
    model_path = args.model
    if model_path is None:
        default_models = [
            "./outputs/models/baseline_B64_E30_L0.001_OAdam_LCrossEntropy.pth",
            "./outputs/models/best_model_resnet50.pth",
            "./outputs/models/final_model_resnet50.pth",
        ]
        for mp in default_models:
            if os.path.exists(mp):
                model_path = mp
                break
    
    if model_path is None or not os.path.exists(model_path):
        print("=" * 60)
        print("ERROR: No trained model found!")
        print("=" * 60)
        print("\nPlease train a model first:")
        print("  python main.py --mode train")
        print("\nOr specify model path:")
        print("  python predict.py --model path/to/model.pth")
        print("=" * 60)
        sys.exit(1)
    
    # Model architecture is resnet50 (only supported architecture)
    model_name = "resnet50"
    
    # Run prediction
    predict_mode(
        input_dir=args.input,
        model_path=model_path,
        output_dir=args.output,
        model_name=model_name,
        save_results=True,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
