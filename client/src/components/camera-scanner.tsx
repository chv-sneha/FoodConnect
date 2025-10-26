import { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Camera, X, Zap, Loader2, RotateCcw, FlashlightIcon } from 'lucide-react';
import { extractTextFromImage, preprocessImageForOCR } from '@/lib/ocr';
import { parseProductName } from '@/lib/ingredient-analyzer';

interface CameraScannerProps {
  onScanComplete: (result: {
    file: File;
    extractedText: string;
    productName: string;
  }) => void;
  onClose: () => void;
}

export function CameraScanner({ onScanComplete, onClose }: CameraScannerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  const [hasFlash, setHasFlash] = useState(false);
  const [flashEnabled, setFlashEnabled] = useState(false);

  // Test div for slide-in-top-normal animation
  // Remove after testing
  const testAnimationDiv = (
    <div className="slide-in-top-normal" style={{ background: '#e0e7ff', color: '#1e293b', padding: '12px', borderRadius: '8px', marginBottom: '16px', textAlign: 'center', fontWeight: 600 }}>
      Slide-in-top-normal Animation Test
    </div>
  );

  const startCamera = useCallback(async () => {
    try {
      // Stop existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const constraints: MediaStreamConstraints = {
        video: {
          facingMode,
          width: { ideal: 1920, min: 1280 },
          height: { ideal: 1080, min: 720 },
          focusMode: { ideal: 'continuous' },
          exposureMode: { ideal: 'continuous' },
          whiteBalanceMode: { ideal: 'continuous' }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // Check for flash support
      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities();
      setHasFlash(!!capabilities.torch);
      setIsScanning(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions.');
    }
  }, [facingMode]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsScanning(false);
  }, []);

  const toggleFlash = useCallback(async () => {
    if (streamRef.current && hasFlash) {
      const track = streamRef.current.getVideoTracks()[0];
      try {
        await track.applyConstraints({
          advanced: [{ torch: !flashEnabled }]
        });
        setFlashEnabled(!flashEnabled);
      } catch (error) {
        console.error('Flash not supported:', error);
      }
    }
  }, [flashEnabled, hasFlash]);

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    setIsProcessing(true);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');

      if (!context) throw new Error('Canvas context not available');

      // Set canvas size to video dimensions
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Apply image enhancement for better OCR
      const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
      const enhancedImageData = enhanceImageQuality(imageData);
      context.putImageData(enhancedImageData, 0, 0);

      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.9);
      });

      const file = new File([blob], 'scanned-image.jpg', { type: 'image/jpeg' });

      // Process with OCR
      const preprocessedFile = await preprocessImageForOCR(file);
      const ocrResult = await extractTextFromImage(preprocessedFile);

      if (ocrResult.confidence < 40) {
        // Try again with different enhancement
        const retryImageData = enhanceImageQualityAgressive(imageData);
        context.putImageData(retryImageData, 0, 0);
        
        const retryBlob = await new Promise<Blob>((resolve) => {
          canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.95);
        });
        const retryFile = new File([retryBlob], 'scanned-image-retry.jpg', { type: 'image/jpeg' });
        const retryPreprocessed = await preprocessImageForOCR(retryFile);
        const retryOcrResult = await extractTextFromImage(retryPreprocessed);
        
        if (retryOcrResult.confidence > ocrResult.confidence) {
          const productName = parseProductName(retryOcrResult.text);
          onScanComplete({
            file: retryPreprocessed,
            extractedText: retryOcrResult.text,
            productName
          });
          stopCamera();
          return;

        }
      }

      const productName = parseProductName(ocrResult.text);
      onScanComplete({
        file: preprocessedFile,
        extractedText: ocrResult.text,
        productName
      });

      stopCamera();
    } catch (error) {
      console.error('Error capturing and analyzing:', error);
      alert('Failed to process image. Please try again.');
      setIsProcessing(false);
    }
  }, [isProcessing, onScanComplete, stopCamera]);

  // Image enhancement functions
  const enhanceImageQuality = (imageData: ImageData): ImageData => {
    const data = imageData.data;
    const enhanced = new ImageData(imageData.width, imageData.height);
    
    for (let i = 0; i < data.length; i += 4) {
      // Increase contrast and brightness
      const r = Math.min(255, Math.max(0, (data[i] - 128) * 1.5 + 128 + 20));
      const g = Math.min(255, Math.max(0, (data[i + 1] - 128) * 1.5 + 128 + 20));
      const b = Math.min(255, Math.max(0, (data[i + 2] - 128) * 1.5 + 128 + 20));
      
      enhanced.data[i] = r;
      enhanced.data[i + 1] = g;
      enhanced.data[i + 2] = b;
      enhanced.data[i + 3] = data[i + 3];
    }
    
    return enhanced;
  };

  const enhanceImageQualityAgressive = (imageData: ImageData): ImageData => {
    const data = imageData.data;
    const enhanced = new ImageData(imageData.width, imageData.height);
    
    for (let i = 0; i < data.length; i += 4) {
      // Convert to grayscale and increase contrast dramatically
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const enhanced_gray = gray > 128 ? Math.min(255, gray * 1.8) : Math.max(0, gray * 0.3);
      
      enhanced.data[i] = enhanced_gray;
      enhanced.data[i + 1] = enhanced_gray;
      enhanced.data[i + 2] = enhanced_gray;
      enhanced.data[i + 3] = data[i + 3];
    }
    
    return enhanced;
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  const switchCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-black text-white">
        <h2 className="text-lg font-semibold">Scan Ingredient Label</h2>
        <Button variant="ghost" size="icon" onClick={onClose} className="text-white">
          <X size={24} />
        </Button>
      </div>

      {/* Camera View */}
      <div className="flex-1 relative">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          autoPlay
          playsInline
          muted
        />
        
        {/* Overlay with scanning frame */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative">
            {/* Scanning frame */}
            <div className="w-80 h-48 border-4 border-primary rounded-xl relative">
              {/* Corner indicators */}
              <div className="absolute -top-2 -left-2 w-8 h-8 border-l-4 border-t-4 border-white rounded-tl-lg"></div>
              <div className="absolute -top-2 -right-2 w-8 h-8 border-r-4 border-t-4 border-white rounded-tr-lg"></div>
              <div className="absolute -bottom-2 -left-2 w-8 h-8 border-l-4 border-b-4 border-white rounded-bl-lg"></div>
              <div className="absolute -bottom-2 -right-2 w-8 h-8 border-r-4 border-b-4 border-white rounded-br-lg"></div>
              
              {/* Scanning line animation */}
              {isScanning && !isProcessing && (
                <div className="absolute inset-0 overflow-hidden rounded-lg">
                  <div className="absolute w-full h-0.5 bg-primary animate-bounce-slow top-4"></div>
                </div>
              )}
            </div>
            
            {/* Instructions */}
            <div className="absolute -bottom-16 left-1/2 transform -translate-x-1/2 text-center">
              <p className="text-white text-sm bg-black bg-opacity-50 px-4 py-2 rounded-lg">
                {isProcessing ? 'Processing...' : 'Position ingredient label in the frame'}
              </p>
            </div>
          </div>
        </div>

        {/* Dark overlay outside scanning area */}
        <div className="absolute inset-0 bg-black bg-opacity-40"></div>
      </div>

      {/* Controls */}
      <div className="bg-black p-6">
        <div className="flex items-center justify-between max-w-sm mx-auto">
          {/* Flash toggle */}
          {hasFlash && (
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleFlash}
              className={`text-white ${flashEnabled ? 'bg-yellow-500' : ''}`}
            >
              <FlashlightIcon size={24} />
            </Button>
          )}

          {/* Capture button */}
          <Button
            size="lg"
            onClick={captureAndAnalyze}
            disabled={isProcessing || !isScanning}
            className="w-20 h-20 rounded-full bg-primary hover:bg-green-600 border-4 border-white"
          >
            {isProcessing ? (
              <Loader2 className="animate-spin" size={32} />
            ) : (
              <Camera size={32} />
            )}
          </Button>

          {/* Switch camera */}
          <Button
            variant="ghost"
            size="icon"
            onClick={switchCamera}
            className="text-white"
            disabled={isProcessing}
          >
            <RotateCcw size={24} />
          </Button>
        </div>

        <div className="text-center mt-4">
          <p className="text-white text-xs opacity-75">
            Tap the circle to scan • Use flash for better results
          </p>
        </div>
      </div>

      {/* Hidden canvas for image processing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}