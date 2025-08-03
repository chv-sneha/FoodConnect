import { useState, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, Upload, X, Loader2, Scan, QrCode } from 'lucide-react';
import { CameraScanner } from './camera-scanner';
import { extractTextFromImage, preprocessImageForOCR } from '@/lib/ocr';
import { parseProductName } from '@/lib/ingredient-analyzer';

interface UploadZoneProps {
  onImageAnalyzed: (result: {
    file: File;
    extractedText: string;
    productName: string;
  }) => void;
  isLoading?: boolean;
}

export function UploadZone({ onImageAnalyzed, isLoading = false }: UploadZoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showCameraScanner, setShowCameraScanner] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelection(files[0]);
    }
  }, []);

  const handleFileSelection = async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setIsProcessing(true);

    try {
      // Preprocess image for better OCR results
      const processedFile = await preprocessImageForOCR(file);
      
      // Extract text using OCR
      const ocrResult = await extractTextFromImage(processedFile);
      
      if (ocrResult.confidence < 60) {
        alert('Image quality is too low. Please try taking a clearer photo.');
        clearSelection();
        return;
      }

      // Parse product name from extracted text
      const productName = parseProductName(ocrResult.text);

      // Call parent callback with results
      onImageAnalyzed({
        file: processedFile,
        extractedText: ocrResult.text,
        productName
      });

    } catch (error) {
      console.error('Error processing image:', error);
      alert('Failed to process image. Please try again.');
      clearSelection();
    } finally {
      setIsProcessing(false);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  const openCameraDialog = () => {
    cameraInputRef.current?.click();
  };

  const openCameraScanner = () => {
    setShowCameraScanner(true);
  };

  const handleCameraScanComplete = (result: {
    file: File;
    extractedText: string;
    productName: string;
  }) => {
    setShowCameraScanner(false);
    onImageAnalyzed(result);
  };

  if (selectedFile && previewUrl) {
    return (
      <div className="bg-white rounded-3xl shadow-xl p-8">
        <div className="relative">
          <img 
            src={previewUrl} 
            alt="Selected product" 
            className="w-full max-h-96 object-contain rounded-2xl"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-2 right-2 rounded-full"
            onClick={clearSelection}
            disabled={isProcessing || isLoading}
          >
            <X size={16} />
          </Button>
        </div>
        
        {isProcessing && (
          <div className="mt-6 flex items-center justify-center space-x-3 text-primary">
            <Loader2 className="animate-spin" size={24} />
            <span className="text-lg font-medium">Analyzing image...</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-3xl shadow-xl p-8">
      <div
        className={`
          border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300
          ${dragActive ? 'border-primary bg-green-50' : 'border-primary hover:border-green-600 hover:bg-green-50'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!isLoading ? openFileDialog : undefined}
      >
        <div className={`mb-6 ${isLoading ? '' : 'animate-pulse'}`}>
          <Camera className="mx-auto text-primary" size={64} />
        </div>
        
        <h3 className="text-2xl font-semibold text-gray-900 mb-3">
          Upload Product Image
        </h3>
        <p className="text-gray-600 mb-6">
          Drag and drop or click to select the back side of your food package
        </p>
        
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Real-time Camera Scan - Primary option */}
          <div className="sm:col-span-3">
            <Button
              type="button"
              className="w-full bg-gradient-to-r from-primary to-green-500 hover:from-green-600 hover:to-green-700 text-white py-4 text-lg font-semibold shadow-lg"
              onClick={(e) => {
                e.stopPropagation();
                openCameraScanner();
              }}
              disabled={isLoading}
            >
              <Camera size={24} className="mr-3" />
              <span>📱 Instant Camera Scan</span>
            </Button>
            <p className="text-center text-xs text-gray-500 mt-2">
              Fast & accurate like PhonePe scanner • Works with any camera quality
            </p>
          </div>

          {/* Alternative options */}
          <Button
            type="button"
            variant="outline"
            className="flex items-center justify-center space-x-2 py-3 px-4"
            onClick={(e) => {
              e.stopPropagation();
              openFileDialog();
            }}
            disabled={isLoading}
          >
            <Upload size={18} />
            <span>Upload Image</span>
          </Button>

          <Button
            type="button"
            variant="outline"
            className="flex items-center justify-center space-x-2 py-3 px-4"
            onClick={(e) => {
              e.stopPropagation();
              openCameraDialog();
            }}
            disabled={isLoading}
          >
            <Camera size={18} />
            <span>Take Photo</span>
          </Button>

          <Button
            type="button"
            variant="outline"
            className="flex items-center justify-center space-x-2 py-3 px-4"
            disabled={true}
          >
            <QrCode size={18} />
            <span>Barcode (Soon)</span>
          </Button>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelection(file);
        }}
      />
      
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelection(file);
        }}
      />

      {/* Camera Scanner Modal */}
      {showCameraScanner && (
        <CameraScanner
          onScanComplete={handleCameraScanComplete}
          onClose={() => setShowCameraScanner(false)}
        />
      )}
    </div>
  );
}
