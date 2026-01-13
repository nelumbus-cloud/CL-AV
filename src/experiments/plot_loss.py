
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse
import os

def plot_training_curves(log_file, output_file):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found. Skipping plot.")
        return

    epochs = []
    difficulties = []
    train_losses = []
    val_losses = []

    print(f"Reading log file: {log_file}")
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # The CSV header is: epoch,difficulty,train_loss,val_loss
                epochs.append(int(row['epoch']))
                difficulties.append(float(row['difficulty']))
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
            except ValueError as e:
                print(f"Skipping malformed row: {row} - {e}")
                continue

    if not epochs:
        print("No valid data found in log file.")
        return

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red', linestyle='-')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:orange', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Difficulty Lambda', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, difficulties, label='Difficulty', color=color, linestyle=':')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.title(f"Training Progress: {os.path.basename(log_file)}")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    print(f"Saving plot to {output_file}")
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, required=True, help='Path to the CSV log file')
    parser.add_argument('--output_file', type=str, default='training_plot.png', help='Path to save the plot')
    
    args = parser.parse_args()
    
    plot_training_curves(args.log_file, args.output_file)
