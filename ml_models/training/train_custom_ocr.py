#!/usr/bin/env python3
"""
Custom OCR Training for Food Labels
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Trainer, TrainingArguments
import cv2

class FoodLabelDataset(Dataset):
    """Dataset for food label OCR training"""
    
    def __init__(self, data_dir: str, annotations_file: str, processor, max_length=256):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess image
        image = self.preprocess_image(image)
        
        # Get text
        text = item['text']
        
        # Process with TrOCR processor
        encoding = self.processor(
            image, 
            text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'labels': encoding['labels'].squeeze()
        }
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply CLAHE for contrast enhancement
        if len(img_array.shape) == 3:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL
        return Image.fromarray(img_array)

class CustomOCRTrainer:
    """Custom trainer for food label OCR"""
    
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Configure model for food labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 256
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
    
    def prepare_datasets(self, train_dir: str, val_dir: str, train_annotations: str, val_annotations: str):
        """Prepare training and validation datasets"""
        
        train_dataset = FoodLabelDataset(
            train_dir, train_annotations, self.processor
        )
        
        val_dataset = FoodLabelDataset(
            val_dir, val_annotations, self.processor
        )
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir: str, epochs: int = 5):
        """Train the custom OCR model"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        return trainer

def create_training_data():
    """Create training data format for food labels"""
    
    # Example annotation format
    annotations = [
        {
            "image": "label1.jpg",
            "text": "Ingredients: Wheat flour, Sugar, Palm oil, Salt, Baking powder",
            "type": "ingredients"
        },
        {
            "image": "label2.jpg", 
            "text": "Energy: 450 kcal, Protein: 8g, Carbohydrates: 65g, Fat: 18g",
            "type": "nutrition"
        },
        {
            "image": "label3.jpg",
            "text": "FSSAI Lic No: 12345678901234",
            "type": "fssai"
        }
    ]
    
    return annotations

def augment_training_data(image_dir: str, output_dir: str):
    """Augment training data with various transformations"""
    
    augmentations = [
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3),
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            
            # Apply augmentations
            for i, aug in enumerate(augmentations):
                augmented = aug(image)
                output_path = os.path.join(output_dir, f"{filename[:-4]}_aug_{i}.jpg")
                augmented.save(output_path)

# Training pipeline
def main():
    """Main training pipeline"""
    
    # Initialize trainer
    trainer = CustomOCRTrainer()
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_datasets(
        train_dir="data/train/images",
        val_dir="data/val/images", 
        train_annotations="data/train/annotations.json",
        val_annotations="data/val/annotations.json"
    )
    
    # Train model
    trained_model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="models/custom_food_ocr",
        epochs=10
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()