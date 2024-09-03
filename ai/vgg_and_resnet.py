import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import copy
from PIL import Image
from sklearn.metrics import recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from torchvision import models
import torch.nn.functional as F

# Load data
data_path = './entry_activity_image_csv_80_10_10_confirmed.csv'  # Update this path
df = pd.read_csv(data_path)

# Parameters
BATCH_SIZE = 16
NUM_CLASSES = len(df['user_id'].unique())
EPOCHS = 100
PATIENCE = 5

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class SpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = dataframe[['label_type']].values
        self.encoder = OneHotEncoder()
        self.labels_encoded = self.encoder.fit_transform(self.labels).toarray()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')  # This returns a torch Tensor

        if self.transform:
            image = self.transform(image)

        label = self.labels_encoded[idx]
        return image, torch.tensor(label, dtype=torch.float32)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss


# Train/Valid split
train_df = df[df['dataset_type'] == 'train']
valid_df = df[df['dataset_type'] == 'valid']

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Change the input size to 224x224
    transforms.ToTensor()
])

# Data loaders
train_dataset = SpectrogramDataset(train_df, transform=transform)
valid_dataset = SpectrogramDataset(valid_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained models
models_dict = {
    'VGG16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    'ResNet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
}

# Modify the classifiers to match the number of classes and add softmax
for name, model in models_dict.items():
    if 'VGG' in name:
        model.classifier[6] = nn.Sequential(
            nn.Linear(model.classifier[6].in_features, NUM_CLASSES),
            nn.Softmax(dim=1)
        )
    else:
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, NUM_CLASSES),
            nn.Softmax(dim=1)
        )
    models_dict[name] = model.to(device)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizers_dict = {
    'VGG16': optim.AdamW(models_dict['VGG16'].parameters(), lr=0.00001),
    'ResNet18': optim.AdamW(models_dict['ResNet18'].parameters(), lr=0.00001)
}

# Early stopping instances
early_stoppings_dict = {
    'VGG16': EarlyStopping(patience=PATIENCE, verbose=True),
    'ResNet18': EarlyStopping(patience=PATIENCE, verbose=True)
}

def plot_and_save_confusion_matrix(conf_matrix, class_names, axis, model_name):
    fig, ax = plt.subplots(figsize=(10, 10))  # Larger figure size for more space
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix for {axis} - {model_name}')
    plt.tight_layout()

    save_path = f'./confusion_matrix_{axis}_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)

def train_model(model, optimizer, train_loader, valid_loader, criterion, early_stopping, axis_label, model_name):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    train_recalls, valid_recalls = [], []
    train_f1s, valid_f1s = [], []
    train_top3_accs, valid_top3_accs = [], []

    class_names = train_loader.dataset.dataframe['label_type'].unique()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS} for {model_name}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            top3_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, torch.max(labels, 1)[1])
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.max(labels, 1)[1])

                # Calculate top-3 accuracy
                top3_preds = torch.topk(outputs, 3, dim=1).indices
                labels_max = torch.max(labels, 1)[1]
                top3_corrects += sum([1 if labels_max[i] in top3_preds[i] else 0 for i in range(len(labels_max))])

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            top3_acc = top3_corrects / len(dataloader.dataset)

            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
                train_recalls.append(recall)
                train_f1s.append(f1)
                train_top3_accs.append(top3_acc)
            else:
                valid_losses.append(epoch_loss)
                valid_accuracies.append(epoch_acc)
                valid_recalls.append(recall)
                valid_f1s.append(f1)
                valid_top3_accs.append(top3_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {recall:.4f} F1: {f1:.4f} Top-3 Acc: {top3_acc:.4f}')

            # Early stopping
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':
                early_stopping(epoch_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        if early_stopping.early_stop:
            print("Exiting loop due to early stopping")
            break

    model.load_state_dict(best_model_wts)

    return model, train_losses, valid_losses, train_accuracies, valid_accuracies, train_recalls, valid_recalls, train_f1s, valid_f1s, train_top3_accs, valid_top3_accs

axes = ['X', 'Y', 'Z']
results = {}

for axis in axes:
    for model_name, model in models_dict.items():
        print(f"Training {model_name} model for {axis} axis data")

        optimizer = optimizers_dict[model_name]
        early_stopping = early_stoppings_dict[model_name]

        # Adjust data loaders for the current axis
        train_df_axis = train_df[train_df['label_type'] == axis]
        valid_df_axis = valid_df[valid_df['label_type'] == axis]

        if train_df_axis.empty or valid_df_axis.empty:
            print(f"No data available for axis {axis}. Skipping training for this axis.")
            continue  # Skip this axis if there's no data

        train_loader = DataLoader(SpectrogramDataset(train_df_axis, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(SpectrogramDataset(valid_df_axis, transform=transform), batch_size=BATCH_SIZE, shuffle=False)

        model, train_losses, valid_losses, train_accuracies, valid_accuracies, train_recalls, valid_recalls, train_f1s, valid_f1s, train_top3_accs, valid_top3_accs = train_model(
            model, optimizer, train_loader, valid_loader, criterion, early_stopping, axis, model_name
        )

        results[f"{model_name}_{axis}"] = {
            "model": model,
            "train_loss": train_losses,
            "valid_loss": valid_losses,
            "train_accuracy": train_accuracies,
            "valid_accuracy": valid_accuracies,
            "train_recall": train_recalls,
            "valid_recall": valid_recalls,
            "train_f1": train_f1s,
            "valid_f1": valid_f1s,
            "train_top3_acc": train_top3_accs,
            "valid_top3_acc": valid_top3_accs
        }

    # Reset the early stopping instances
    early_stoppings_dict = {
        'VGG16': EarlyStopping(patience=PATIENCE, verbose=True),
        'ResNet18': EarlyStopping(patience=PATIENCE, verbose=True)
    }


# Plot and save the metrics
def plot_and_save_metrics(results, metric_name):
    for axis in axes:
        plt.figure(figsize=(10, 6))

        for model_name in models_dict.keys():
            key = f"{model_name}_{axis}"
            if key not in results:
                print(f"Skipping plot for {model_name} on axis {axis} due to missing results.")
                continue

            train_metric = results[key][f"train_{metric_name.lower()}"]
            valid_metric = results[key][f"valid_{metric_name.lower()}"]
            
            # Ensure train_metric and valid_metric are converted to numpy arrays
            train_metric = np.array(train_metric)
            valid_metric = np.array(valid_metric)
            
            plt.plot(train_metric, label=f'{model_name} Training {metric_name}')
            plt.plot(valid_metric, label=f'{model_name} Validation {metric_name}')

        plt.title(f'{metric_name} for Axis {axis}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = f'./{metric_name.lower()}_{axis}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"{metric_name} plot saved to: {save_path}")
        plt.close()


# Save all metrics
metrics = ["Loss", "Accuracy", "Recall", "F1", "Top3_Accuracy"]
for metric in metrics:
    plot_and_save_metrics(results, metric)

# Sample code to load a saved model and continue training if needed
# for model_name in models_dict.keys():
#     for axis in axes:
#         checkpoint_path = f'./best_model_{model_name}_{axis}.pth'
#         model.load_state_dict(torch.load(checkpoint_path))
#         # You can add more training here if necessary
