#!/usr/bin/env python3
"""
Product Name Extraction Model - CNN+LSTM Architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import json

class ProductNameDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get bounding box for product name
        bbox = eval(row['product_name_bbox'])  # [x1, y1, x2, y2]
        
        return {
            'image': image,
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'product_name': row['product_name']
        }

class ProductNameCNN(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=128, hidden_dim=256):
        super(ProductNameCNN, self).__init__()
        
        # CNN for image feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # x1, y1, x2, y2
        )
        
        # Text classification head
        self.text_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )
        
    def forward(self, x):
        # Extract CNN features
        features = self.cnn(x)
        features_flat = features.view(features.size(0), -1)
        
        # Predict bounding box
        bbox_pred = self.bbox_head(features_flat)
        
        # Predict text class
        text_pred = self.text_head(features_flat)
        
        return bbox_pred, text_pred

class ProductNameTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.bbox_criterion = nn.MSELoss()
        self.text_criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            bbox_targets = batch['bbox'].to(self.device)
            
            self.optimizer.zero_grad()
            
            bbox_pred, text_pred = self.model(images)
            
            # Calculate losses
            bbox_loss = self.bbox_criterion(bbox_pred, bbox_targets)
            
            # Combined loss
            loss = bbox_loss
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        bbox_accuracies = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                bbox_targets = batch['bbox'].to(self.device)
                
                bbox_pred, text_pred = self.model(images)
                
                bbox_loss = self.bbox_criterion(bbox_pred, bbox_targets)
                total_loss += bbox_loss.item()
                
                # Calculate IoU for bbox accuracy
                iou = self.calculate_iou(bbox_pred, bbox_targets)
                bbox_accuracies.extend(iou.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        avg_bbox_acc = np.mean(bbox_accuracies)
        
        return avg_loss, avg_bbox_acc
    
    def calculate_iou(self, pred_boxes, target_boxes):
        """Calculate Intersection over Union for bounding boxes"""
        # Convert to [x1, y1, x2, y2] format
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(1)
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        return iou

def create_training_data():
    """Create training dataset for product name extraction"""
    
    # Sample training data structure
    training_data = [
        {
            'image_path': 'data/labels/label1.jpg',
            'product_name': 'Maggi Noodles',
            'product_name_bbox': '[50, 20, 200, 60]',  # x1, y1, x2, y2
            'ingredients_bbox': '[30, 100, 250, 150]',
            'nutrition_bbox': '[30, 200, 250, 300]'
        },
        {
            'image_path': 'data/labels/label2.jpg',
            'product_name': 'Parle Biscuits',
            'product_name_bbox': '[40, 15, 180, 50]',
            'ingredients_bbox': '[25, 80, 220, 120]',
            'nutrition_bbox': '[25, 180, 220, 280]'
        }
    ]
    
    return pd.DataFrame(training_data)

def train_product_name_model():
    """Main training function"""
    
    # Create dataset
    df = create_training_data()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = ProductNameDataset(train_df, transform=transform)
    val_dataset = ProductNameDataset(val_df, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProductNameCNN()
    trainer = ProductNameTrainer(model, device)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(50):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        print(f'Epoch {epoch+1}/50:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/product_name_model.pth')
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model

if __name__ == "__main__":
    model = train_product_name_model()