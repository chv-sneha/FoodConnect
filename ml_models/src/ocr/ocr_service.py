"""
OCR Service for Food Label Text Extraction
Supports multiple OCR engines with fallback mechanism
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
import base64
import json
from typing import Dict, List, Optional, Tuple
import re

class OCRService:
    def __init__(self, google_api_key: Optional[str] = None):
        """
        Initialize OCR Service with optional Google Vision API key
        
        Args:
            google_api_key: Google Cloud Vision API key (optional)
        """
        self.google_api_key = google_api_key
        self.confidence_threshold = 60  # Minimum confidence for Tesseract
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_tesseract(self, image_path: str) -> Dict:
        """
        Extract text using Tesseract OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted text and confidence
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()%/-: '
            
            # Extract text with confidence
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence text
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            texts = [data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 30]
            
            # Combine text
            extracted_text = ' '.join(texts).strip()
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': extracted_text,
                'confidence': avg_confidence,
                'method': 'tesseract',
                'success': True
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'method': 'tesseract',
                'success': False,
                'error': str(e)
            }
    
    def extract_text_google_vision(self, image_path: str) -> Dict:
        """
        Extract text using Google Vision API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted text and confidence
        """
        if not self.google_api_key:
            return {
                'text': '',
                'confidence': 0,
                'method': 'google_vision',
                'success': False,
                'error': 'Google API key not provided'
            }
        
        try:
            # Encode image to base64
            with open(image_path, 'rb') as image_file:
                image_content = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare API request
            url = f'https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}'
            
            payload = {
                'requests': [{
                    'image': {'content': image_content},
                    'features': [{'type': 'TEXT_DETECTION'}]
                }]
            }
            
            # Make API call
            response = requests.post(url, json=payload)
            result = response.json()
            
            if 'responses' in result and result['responses']:
                annotations = result['responses'][0].get('textAnnotations', [])
                
                if annotations:
                    # First annotation contains full text
                    full_text = annotations[0]['description']
                    
                    # Calculate average confidence
                    confidences = []
                    for annotation in annotations[1:]:  # Skip first (full text)
                        if 'confidence' in annotation:
                            confidences.append(annotation['confidence'] * 100)
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 95
                    
                    return {
                        'text': full_text,
                        'confidence': avg_confidence,
                        'method': 'google_vision',
                        'success': True
                    }
            
            return {
                'text': '',
                'confidence': 0,
                'method': 'google_vision',
                'success': False,
                'error': 'No text detected'
            }
            
        except Exception as e:
            return {
                'text': '',
                'confidence': 0,
                'method': 'google_vision',
                'success': False,
                'error': str(e)
            }
    
    def extract_text(self, image_path: str) -> Dict:
        """
        Extract text using hybrid approach (Tesseract + Google Vision fallback)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        # Try Tesseract first
        tesseract_result = self.extract_text_tesseract(image_path)
        
        # If Tesseract confidence is high enough, use it
        if tesseract_result['success'] and tesseract_result['confidence'] >= self.confidence_threshold:
            return tesseract_result
        
        # Otherwise, try Google Vision API
        if self.google_api_key:
            google_result = self.extract_text_google_vision(image_path)
            
            # If Google Vision succeeds, use it
            if google_result['success']:
                return google_result
        
        # If both fail or Google API not available, return Tesseract result anyway
        return tesseract_result
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('0', 'O', 1)  # First occurrence only
        text = text.replace('5', 'S', 1)  # First occurrence only
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s.,()%/-:]', '', text)
        
        return text.strip()

# Example usage and testing
if __name__ == "__main__":
    # Initialize OCR service
    ocr = OCRService()
    
    # Test with sample image (you'll need to provide actual image path)
    # result = ocr.extract_text('path/to/food/label/image.jpg')
    # print(f"Extracted text: {result['text']}")
    # print(f"Confidence: {result['confidence']}")
    # print(f"Method: {result['method']}")
    
    print("OCR Service initialized successfully!")