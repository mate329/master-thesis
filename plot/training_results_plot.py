import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_file_1, csv_file_2):
    # Load the data from the CSV files
    df1 = pd.read_csv(csv_file_1)
    df2 = pd.read_csv(csv_file_2)

    # Extract epochs
    epochs1 = df1['Epoch']
    epochs2 = df2['Epoch']

    # List of metrics to plot
    metrics = [
        'Train Loss', 'Valid Loss',
        'Train Accuracy', 'Valid Accuracy',
        'Train Recall', 'Valid Recall',
        'Train F1 Score', 'Valid F1 Score'
    ]

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        
        plt.plot(epochs1, df1[metric], label=f'VGG Unos PIN-a {metric}', marker='o')
        plt.plot(epochs2, df2[metric], label=f'VGG Unos teksta {metric}', marker='x')
        
        plt.xlabel('Epohe')
        plt.ylabel(metric)
        plt.title(f'{metric} usporedba')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        # plt.show()
        plt.savefig(f'{metric}_comparison.png')

# Example usage
csv_file_1 = '/home/matia/dev/znanstveni/ai/enterpin_results/enterpin_metrics_x-axis_vgg16_top1.csv'
csv_file_2 = '/home/matia/dev/znanstveni/ai/entryactivity_results/training_results_csv/metrics_x-axis_vgg16.csv'

plot_metrics(csv_file_1, csv_file_2)
