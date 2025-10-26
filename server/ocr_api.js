// OCR API endpoint for your website
const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `food_label_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.${file.originalname.split('.').pop()}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|bmp|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'));
    }
  }
});

// OCR endpoint
router.post('/analyze-image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const imagePath = req.file.path;
    console.log(`📸 Processing image: ${imagePath}`);

    // Call Python OCR script
    const pythonProcess = spawn('python', [
      path.join(__dirname, '../ml_models/analyze_image.py'),
      imagePath
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      // Clean up uploaded file
      fs.unlink(imagePath, (err) => {
        if (err) console.log('Error deleting temp file:', err);
      });

      if (code === 0) {
        try {
          const ocrResult = JSON.parse(result);
          res.json({
            success: true,
            extractedText: ocrResult.text,
            confidence: ocrResult.confidence,
            method: ocrResult.method
          });
        } catch (parseError) {
          res.status(500).json({ 
            error: 'Failed to parse OCR result',
            details: parseError.message 
          });
        }
      } else {
        res.status(500).json({ 
          error: 'OCR processing failed',
          details: error 
        });
      }
    });

  } catch (error) {
    console.error('OCR API Error:', error);
    res.status(500).json({ 
      error: 'Internal server error',
      details: error.message 
    });
  }
});

module.exports = router;