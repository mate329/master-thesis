import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# Run the processing
root_directory = './entryactivity_results'  # Replace with your root directory path
csv_file_path = './entry_activity_image_csv_80_10_10_confirmed.csv'  # Replace with your CSV file path

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(csv_file_path)
NUM_CLASSES = len(df['user_id'].unique())

# Custom dataset for loading images from paths in CSV
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['user_id']

        if self.transform:
            image = self.transform(image)

        return image, label

# Load test dataset based on model type
def get_test_loader(csv_file, label_type, batch_size=16):
    df = pd.read_csv(csv_file)
    test_df = df[(df['dataset_type'] == 'test') & (df['label_type'] == label_type)]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(test_df, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Function to load and initialize model
def initialize_model(model_type, num_classes):
    if 'vgg' in model_type.lower():
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    elif 'resnet' in model_type.lower():
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError("Unsupported model type")

    model = model.to(DEVICE)
    return model

# Evaluate the model on the test dataset
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Process models recursively in directories
def process_models_recursively(root_dir, csv_file):
    results = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pth"):
                pth_file_path = os.path.join(root, file)
                print(f"Processing: {pth_file_path}")

                # Determine the model type (VGG or ResNet)
                if 'vgg' in file.lower():
                    model_type = 'vgg'
                    label_type = 'X'  # Example: Assign label_type 'X' to VGG
                elif 'resnet' in file.lower():
                    model_type = 'resnet'
                    label_type = 'Y'  # Example: Assign label_type 'Y' to ResNet
                else:
                    print(f"Unknown model type in {file}. Skipping...")
                    continue

                num_classes = len(pd.read_csv(csv_file)['user_id'].unique())

                # Initialize model
                model = initialize_model(model_type, num_classes)

                # Load model weights
                try:
                    model.load_state_dict(torch.load(pth_file_path, map_location=DEVICE))
                except RuntimeError as e:
                    print(f"Error loading state dict: {e}")
                    continue

                # Get the test data loader
                test_loader = get_test_loader(csv_file, label_type)

                # Evaluate model
                accuracy = evaluate_model(model, test_loader)
                print(f"Model: {file}, Label Type: {label_type}, Accuracy: {accuracy}")

                # Save the results
                results.append(f"Model: {file}, Label Type: {label_type}, Accuracy: {accuracy:.4f}")

    # Save results to a file
    with open('test_results.txt', 'w') as f:
        for result in results:
            f.write(f"{result}\n")



process_models_recursively(root_directory, csv_file_path)
