const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const router = express.Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ 
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

router.post('/generic', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        success: false, 
        error: 'No image file provided' 
      });
    }

    const imagePath = req.file.path;
    const pythonScript = path.join(__dirname, '../../ml_models/simple_analyze.py');

    const python = spawn('python3', [pythonScript, imagePath]);
    
    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    python.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    python.on('close', (code) => {
      fs.unlink(imagePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });

      if (code !== 0) {
        console.error('Python script error:', errorString);
        return res.status(500).json({
          success: false,
          error: 'Analysis failed. Please try again.'
        });
      }

      try {
        const result = JSON.parse(dataString);
        result.timestamp = new Date().toISOString();
        result.analysis_type = 'generic';
        res.json(result);
      } catch (parseError) {
        console.error('JSON parse error:', parseError);
        res.status(500).json({
          success: false,
          error: 'Failed to process analysis results'
        });
      }
    });

  } catch (error) {
    console.error('Analysis error:', error);
    
    if (req.file && req.file.path) {
      fs.unlink(req.file.path, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
    }
    
    res.status(500).json({
      success: false,
      error: 'Internal server error during analysis'
    });
  }
});

router.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'Analysis service is running',
    timestamp: new Date().toISOString()
  });
});

module.exports = router;