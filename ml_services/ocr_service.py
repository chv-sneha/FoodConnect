#!/usr/bin/env python3
"""
Production OCR Service - Hybrid Cloud + Local Fallback
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import requests
import base64
from typing import Dict, List, Optional, Tuple
import logging

class ProductionOCR:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_confidence = 60
        self.high_confidence = 85
        
    def preprocess_image(self, image_path: str) -> str:
        """Advanced image preprocessing for better OCR accuracy"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # 1. Orientation correction
            img = self._correct_orientation(img)
            
            # 2. Enhance contrast and sharpness
            img = self._enhance_image(img)
            
            # 3. Adaptive threshold
            processed_img = self._adaptive_threshold(img)
            
            # Save processed image
            processed_path = image_path.replace('.', '_processed.')
            cv2.imwrite(processed_path, processed_img)
            
            return processed_path
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return image_path
    
    def _correct_orientation(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct image orientation"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        best_score = 0
        best_rotation = 0
        
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = img
            else:
                center = (img.shape[1]//2, img.shape[0]//2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
            
            gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            try:
                text = pytesseract.image_to_string(gray_rot, config='--psm 6')
                score = len([c for c in text if c.isalnum()])
                if score > best_score:
                    best_score = score
                    best_rotation = angle
            except:
                continue
        
        if best_rotation != 0:
            center = (img.shape[1]//2, img.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, best_rotation, 1.0)
            img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        
        return img
    
    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR"""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for text regions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text_google_vision(self, image_path: str) -> Dict:
        """Extract text using Google Vision API"""
        try:
            # Check if Google Vision is available
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                return {'success': False, 'error': 'Google Vision credentials not found'}
            
            from google.cloud import vision
            
            client = vision.ImageAnnotatorClient()
            
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            
            if response.error.message:
                return {'success': False, 'error': response.error.message}
            
            text_blocks = []
            full_text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            block_text += word_text + " "
                    
                    if block_text.strip():
                        text_blocks.append({
                            'text': block_text.strip(),
                            'confidence': 90,
                            'source': 'google_vision'
                        })
            
            return {
                'success': True,
                'text_blocks': text_blocks,
                'full_text': full_text,
                'confidence': 90,
                'method': 'google_vision'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_text_tesseract(self, image_path: str) -> Dict:
        """Extract text using Tesseract"""
        try:
            config = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(image_path, config=config)
            
            data = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'success': True,
                'text_blocks': [{'text': text, 'confidence': avg_confidence, 'source': 'tesseract'}],
                'full_text': text,
                'confidence': avg_confidence,
                'method': 'tesseract'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_text_hybrid(self, image_path: str) -> Dict:
        """Hybrid OCR: Try cloud first, fallback to local"""
        
        # Preprocess image
        processed_path = self.preprocess_image(image_path)
        
        # Try Google Vision first
        result = self.extract_text_google_vision(processed_path)
        if result['success'] and result['confidence'] > self.high_confidence:
            return result
        
        # Fallback to Tesseract
        result = self.extract_text_tesseract(processed_path)
        return result
    
    def detect_barcode(self, image_path: str) -> Optional[str]:
        """Detect barcode using pyzbar"""
        try:
            from pyzbar import pyzbar
            
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            barcodes = pyzbar.decode(gray)
            
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                if len(barcode_data) in [12, 13, 14]:
                    return barcode_data
            
            return None
            
        except ImportError:
            return None
        except Exception as e:
            return None