"""Entry point: generate data, benchmark models, and save the best one."""
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from ml.trainer import train_and_benchmark

if __name__ == "__main__":
    train_and_benchmark(output_dir="trained_models")
