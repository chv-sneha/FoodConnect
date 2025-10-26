import express from 'express';
import multer from 'multer';
import { spawn } from 'child_process';
import path from 'path';
import cors from 'cors';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json());
app.use(express.static('client/dist'));

// Simple generic analysis endpoint
app.post('/api/generic/analyze', upload.single('image'), (req, res) => {
  console.log('📸 Generic analysis request received');
  
  if (!req.file) {
    return res.status(400).json({ 
      success: false, 
      error: 'No image uploaded' 
    });
  }

  console.log('🔍 Processing image:', req.file.path);

  // Run Python script
  const pythonProcess = spawn('python3', [
    'ml_models/simple_analyze.py',
    req.file.path
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
    console.log('🐍 Python process finished with code:', code);
    
    if (code === 0) {
      try {
        const analysisResult = JSON.parse(result);
        console.log('✅ Analysis successful');
        res.json(analysisResult);
      } catch (parseError) {
        console.log('❌ JSON parse error:', parseError.message);
        console.log('Raw result:', result.substring(0, 200));
        res.status(500).json({
          success: false,
          error: 'Failed to parse analysis result',
          details: parseError.message
        });
      }
    } else {
      console.log('❌ Python process failed:', error);
      res.status(500).json({
        success: false,
        error: 'Analysis process failed',
        details: error
      });
    }
  });
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Simple server running' });
});

// Serve frontend
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'client/dist/index.html'));
});

const PORT = 8080;
app.listen(PORT, () => {
  console.log(`🚀 Simple server running on port ${PORT}`);
  console.log(`🌐 Access at: http://localhost:${PORT}`);
  console.log(`📸 Generic analysis ready at: /api/generic/analyze`);
});