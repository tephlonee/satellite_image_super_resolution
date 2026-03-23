
import os
import json
from datetime import datetime


import matplotlib.pyplot as plt


from utils.logger import get_logger


logger = get_logger(__name__)


def save_training_history(history, output_dir):
  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, f"training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to: {history_path}")

def plot_training_history(history, output_dir):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    epochs = history["epoch"]
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SRCNN Training Loss")
    plt.legend()

    # PSNR plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_psnr"], label="Train PSNR")
    plt.plot(epochs, history["val_psnr"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("SRCNN PSNR")
    plt.legend()

    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path)
    logger.info(f"Training history plot saved to: {plot_path}")


def save_train_info(results, output_dir):
    save_training_history(results["history"], output_dir)
    plot_training_history(results["history"], output_dir)