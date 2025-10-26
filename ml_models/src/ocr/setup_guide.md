# OCR Service Setup Guide

## Step 1: Install Tesseract OCR Engine

### macOS:
```bash
brew install tesseract
```

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Windows:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH

## Step 2: Install Python Dependencies

```bash
cd ml_models
pip install -r requirements_ocr.txt
```

## Step 3: Test OCR Service

```python
from src.ocr.ocr_service import OCRService

# Initialize OCR service
ocr = OCRService()

# Test with image
result = ocr.extract_text('path/to/food/label.jpg')
print(result)
```

## Step 4: Optional - Google Vision API Setup

1. Go to Google Cloud Console
2. Create new project or select existing
3. Enable Vision API
4. Create API key
5. Use in OCR service:

```python
ocr = OCRService(google_api_key="your_api_key_here")
```

## Troubleshooting

### Tesseract not found:
- Check if tesseract is in PATH
- Set custom path: `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

### Low accuracy:
- Ensure good image quality
- Try different preprocessing settings
- Use Google Vision API for complex images