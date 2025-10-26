#!/usr/bin/env python3
"""
Training Pipeline for Custom Food Label Model
Combines OCR + NER + Table Extraction for Indian Food Labels
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class FoodLabelDataset(Dataset):
    """Dataset for food label images with annotations"""
    
    def __init__(self, data_dir: str, annotations_file: str):
        self.data_dir = data_dir
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))  # Standardize size
        image = image.astype(np.float32) / 255.0
        
        # Get annotations
        annotation = self.annotations[image_file]
        
        return {
            'image': torch.tensor(image).permute(2, 0, 1),  # CHW format
            'product_name': annotation.get('product_name', ''),
            'ingredients': annotation.get('ingredients', []),
            'nutrition': annotation.get('nutrition', {}),
            'fssai': annotation.get('fssai', ''),
            'brand': annotation.get('brand', '')
        }

class FoodLabelModel(nn.Module):
    """Multi-task model for food label analysis"""
    
    def __init__(self, num_classes=1000):
        super(FoodLabelModel, self).__init__()
        
        # CNN backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Feature dimension
        feature_dim = 512 * 8 * 8
        
        # Task-specific heads
        self.ingredient_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)  # Multi-label classification
        )
        
        self.nutrition_regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # 7 nutrition values
        )
        
        self.text_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Text/No-text classification
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Task predictions
        ingredient_logits = self.ingredient_classifier(features)
        nutrition_values = self.nutrition_regressor(features)
        text_presence = self.text_detector(features)
        
        return {
            'ingredients': ingredient_logits,
            'nutrition': nutrition_values,
            'text_presence': text_presence
        }

class FoodLabelTrainer:
    """Training pipeline for food label model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.ingredient_loss = nn.BCEWithLogitsLoss()
        self.nutrition_loss = nn.MSELoss()
        self.text_loss = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses (simplified - you'd need proper label encoding)
            loss = 0
            
            # In practice, you'd convert text annotations to proper tensors
            # This is a simplified example
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def create_training_data():
    """
    Instructions for creating training data:
    
    1. Collect 1000+ Indian food label images
    2. Annotate them with this format:
    
    annotations.json:
    {
        "image1.jpg": {
            "product_name": "Britannia Good Day Cookies",
            "ingredients": ["wheat flour", "sugar", "palm oil", "butter"],
            "nutrition": {
                "energy": 456,
                "protein": 6.8,
                "carbohydrates": 70.2,
                "fat": 16.2,
                "fiber": 2.1,
                "sugar": 18.5,
                "sodium": 312
            },
            "fssai": "12345678901234",
            "brand": "Britannia"
        }
    }
    
    3. Use tools like LabelImg or CVAT for annotation
    4. Focus on popular Indian brands: Britannia, Parle, Amul, Maggi, etc.
    """
    
    print("📋 Training Data Creation Guide:")
    print("1. Collect 1000+ food label images")
    print("2. Annotate with product name, ingredients, nutrition, FSSAI")
    print("3. Use the annotation format shown in the code")
    print("4. Focus on Indian brands for better local accuracy")

def main():
    """Main training function"""
    
    # Check if training data exists
    data_dir = "training_data/images"
    annotations_file = "training_data/annotations.json"
    
    if not os.path.exists(data_dir) or not os.path.exists(annotations_file):
        print("⚠️ Training data not found!")
        create_training_data()
        return
    
    # Create dataset and dataloader
    dataset = FoodLabelDataset(data_dir, annotations_file)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model and trainer
    model = FoodLabelModel()
    trainer = FoodLabelTrainer(model)
    
    # Training loop
    print("🚀 Starting training...")
    for epoch in range(50):
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/50, Loss: {loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_model(f"models/food_model_epoch_{epoch+1}.pth")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()