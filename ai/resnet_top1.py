import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import copy
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from torchvision import models  # Import the models

# Load data
# data_path = './entry_activity_image_csv_80_10_10_confirmed.csv'  # Update this path
data_path = './enterpinactivity_image_csv.csv'
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
    def __init__(self, patience=7, verbose=False, delta=0):
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
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
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

# Load a pre-trained ResNet model
model_x = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_y = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_z = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final layer to match the number of classes
model_x.fc = nn.Linear(model_x.fc.in_features, NUM_CLASSES)
model_y.fc = nn.Linear(model_y.fc.in_features, NUM_CLASSES)
model_z.fc = nn.Linear(model_z.fc.in_features, NUM_CLASSES)

# Move models to device
model_x = model_x.to(device)
model_y = model_y.to(device)
model_z = model_z.to(device)

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_x = optim.AdamW(model_x.parameters(), lr=0.00001)
optimizer_y = optim.AdamW(model_y.parameters(), lr=0.00001)
optimizer_z = optim.AdamW(model_z.parameters(), lr=0.00001)

# Early stopping
early_stopping_x = EarlyStopping(patience=PATIENCE, verbose=True)
early_stopping_y = EarlyStopping(patience=PATIENCE, verbose=True)
early_stopping_z = EarlyStopping(patience=PATIENCE, verbose=True)

def plot_and_save_confusion_matrix(conf_matrix, class_names, axis):
    fig, ax = plt.subplots(figsize=(10, 10))  # Larger figure size for more space
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix for Axis {axis}')
    plt.tight_layout()

    save_path = f'./confusion_matrix_{axis}_resnet.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to: {save_path}")

    plt.close(fig)

def train_model(model, optimizer, train_loader, valid_loader, criterion, early_stopping, axis_label):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    train_recalls, valid_recalls = [], []
    train_f1s, valid_f1s = [], []

    class_names = train_loader.dataset.dataframe['label_type'].unique()

    for epoch in range(EPOCHS + 1):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
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

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                train_recalls.append(recall)
                train_f1s.append(f1)
            else:
                valid_losses.append(epoch_loss)
                valid_accuracies.append(epoch_acc.item())
                valid_recalls.append(recall)
                valid_f1s.append(f1)

            print(f'{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping for {axis_label}")
                    break

        if early_stopping.early_stop:
            break

        if phase == 'train':
            conf_matrix = confusion_matrix(all_labels, all_preds)
            plot_and_save_confusion_matrix(conf_matrix, class_names, axis_label)

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies, train_recalls, valid_recalls, train_f1s, valid_f1s

# Train models for each axis
model_x, train_losses_x, valid_losses_x, train_accuracies_x, valid_accuracies_x, train_recalls_x, valid_recalls_x, train_f1s_x, valid_f1s_x = train_model(
    model_x, optimizer_x, train_loader, valid_loader, criterion, early_stopping_x, 'X-axis'
)
model_y, train_losses_y, valid_losses_y, train_accuracies_y, valid_accuracies_y, train_recalls_y, valid_recalls_y, train_f1s_y, valid_f1s_y = train_model(
    model_y, optimizer_y, train_loader, valid_loader, criterion, early_stopping_y, 'Y-axis'
)
model_z, train_losses_z, valid_losses_z, train_accuracies_z, valid_accuracies_z, train_recalls_z, valid_recalls_z, train_f1s_z, valid_f1s_z = train_model(
    model_z, optimizer_z, train_loader, valid_loader, criterion, early_stopping_z, 'Z-axis'
)

# Save models
torch.save(model_x.state_dict(), 'enterpin_resnet_top1_model_x.pth')
torch.save(model_y.state_dict(), 'enterpin_resnet_top1_model_y.pth')
torch.save(model_z.state_dict(), 'enterpin_resnet_top1_model_z.pth')

def save_arrays_to_csv(axis_label, train_losses, valid_losses, train_accuracies, valid_accuracies, train_recalls, valid_recalls, train_f1s, valid_f1s):
    save_dict = {
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Train Accuracy': train_accuracies,
        'Valid Accuracy': valid_accuracies,
        'Train Recall': train_recalls,
        'Valid Recall': valid_recalls,
        'Train F1 Score': train_f1s,
        'Valid F1 Score': valid_f1s
    }

    df = pd.DataFrame(save_dict)
    save_path = f'./enterpin_metrics_{axis_label.lower()}_resnet.csv'
    df.to_csv(save_path, index=False)
    print(f'Saved metrics to: {save_path}')

# Save metrics arrays to CSV
save_arrays_to_csv('X-axis', train_losses_x, valid_losses_x, train_accuracies_x, valid_accuracies_x, train_recalls_x, valid_recalls_x, train_f1s_x, valid_f1s_x)
save_arrays_to_csv('Y-axis', train_losses_y, valid_losses_y, train_accuracies_y, valid_accuracies_y, train_recalls_y, valid_recalls_y, train_f1s_y, valid_f1s_y)
save_arrays_to_csv('Z-axis', train_losses_z, valid_losses_z, train_accuracies_z, valid_accuracies_z, train_recalls_z, valid_recalls_z, train_f1s_z, valid_f1s_z)

print("Training complete!")

def plot_metrics(axis_label, train_losses, valid_losses, train_accuracies, valid_accuracies, train_recalls, valid_recalls, train_f1s, valid_f1s):
    epochs = range(1, len(train_losses) + 1)
    
    save_dir = './plots_resnet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, valid_losses, 'r', label='Validation Loss')
    plt.title(f'{axis_label} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enterpin_{axis_label}_loss_resnet.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, 'r', label='Validation Accuracy')
    plt.title(f'{axis_label} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enterpin_{axis_label}_accuracy_resnet.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_recalls, 'b', label='Training Recall')
    plt.plot(epochs, valid_recalls, 'r', label='Validation Recall')
    plt.title(f'{axis_label} Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enterpin_{axis_label}_recall_resnet.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1s, 'b', label='Training F1 Score')
    plt.plot(epochs, valid_f1s, 'r', label='Validation F1 Score')
    plt.title(f'{axis_label} F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'enterpin_{axis_label}_f1_score_resnet.png'), dpi=300)
    plt.close()

    print(f'Plots saved in directory: {save_dir}')

# Plot metrics for each axis
plot_metrics('X-axis', train_losses_x, valid_losses_x, train_accuracies_x, valid_accuracies_x, train_recalls_x, valid_recalls_x, train_f1s_x, valid_f1s_x)
plot_metrics('Y-axis', train_losses_y, valid_losses_y, train_accuracies_y, valid_accuracies_y, train_recalls_y, valid_recalls_y, train_f1s_y, valid_f1s_y)
plot_metrics('Z-axis', train_losses_z, valid_losses_z, train_accuracies_z, valid_accuracies_z, train_recalls_z, valid_recalls_z, train_f1s_z, valid_f1s_z)
