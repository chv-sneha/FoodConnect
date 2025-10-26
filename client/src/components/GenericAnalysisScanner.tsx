import { useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Camera, Upload, Scan, Volume2, Download, Zap, Shield, Brain } from 'lucide-react';
import { useStore } from '@/store/useStore';

interface ScannerProps {
  onAnalysisComplete: (result: any) => void;
}

export default function GenericAnalysisScanner({ onAnalysisComplete }: ScannerProps) {
  const [isScanning, setIsScanning] = useState(false);
  const [isCameraMode, setIsCameraMode] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraMode(true);
        setIsScanning(true);
        
        // Auto-scan every 2 seconds
        const interval = setInterval(() => {
          if (!isAnalyzing) captureAndAnalyze();
        }, 2000);
        
        return () => clearInterval(interval);
      }
    } catch (error) {
      console.error('Camera access failed:', error);
      alert('Camera access denied. Please use upload mode.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsCameraMode(false);
    setIsScanning(false);
  };

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isAnalyzing) return;
    
    setIsAnalyzing(true);
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx?.drawImage(video, 0, 0);
    
    canvas.toBlob(async (blob) => {
      if (blob) await analyzeImage(blob);
      setIsAnalyzing(false);
    }, 'image/jpeg', 0.8);
  }, [isAnalyzing]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) analyzeImage(file);
  };

  const analyzeImage = async (imageBlob: Blob) => {
    setIsAnalyzing(true);
    console.log('Starting analysis for image:', imageBlob.size, 'bytes');
    
    // Show global loading
    const { setLoading } = useStore.getState();
    setLoading(true);
    
    const formData = new FormData();
    formData.append('image', imageBlob, 'image.jpg');

    try {
      console.log('Sending request to /api/generic/analyze');
      const response = await fetch('/api/generic/analyze', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (response.ok) {
        const result = await response.json();
        console.log('Analysis result:', result);
        
        // Add to recent analyses
        const { addAnalysis } = useStore.getState();
        addAnalysis(result);
        
        onAnalysisComplete(result);
        if (isCameraMode) stopCamera();
      } else {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        console.error('Server error:', errorData);
        throw new Error(errorData.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert(`Analysis failed: ${error.message}. Please try again.`);
    } finally {
      setIsAnalyzing(false);
      const { setLoading } = useStore.getState();
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center justify-center space-x-2">
          <Scan className="text-primary" size={24} />
          <span>Generic Food Analysis</span>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {!isCameraMode ? (
          <div className="space-y-4">
            {/* Enhanced Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <Button 
                onClick={startCamera}
                className="h-40 flex flex-col items-center justify-center space-y-3 bg-gradient-to-br from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white"
                disabled={isAnalyzing}
              >
                <div className="flex items-center gap-2">
                  <Camera size={32} />
                  <Zap size={20} className="text-yellow-300" />
                </div>
                <span className="font-semibold">AI Instant Scan</span>
                <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                  📱 Live Camera + OCR
                </Badge>
              </Button>
              
              <Button 
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="h-40 flex flex-col items-center justify-center space-y-3 border-2 border-dashed border-gray-300 hover:border-primary hover:bg-primary/5"
                disabled={isAnalyzing}
              >
                <div className="flex items-center gap-2">
                  <Upload size={32} />
                  <Brain size={20} className="text-purple-500" />
                </div>
                <span className="font-semibold">Smart Upload</span>
                <Badge variant="outline" className="border-purple-200 text-purple-700">
                  🗪 ML Analysis
                </Badge>
              </Button>
            </div>
            
            {/* Feature Highlights */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <Shield className="mx-auto mb-2 text-green-600" size={24} />
                <p className="text-xs font-medium text-green-800">FSSAI Verified</p>
              </div>
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <Brain className="mx-auto mb-2 text-blue-600" size={24} />
                <p className="text-xs font-medium text-blue-800">AI Powered</p>
              </div>
              <div className="text-center p-3 bg-purple-50 rounded-lg">
                <Zap className="mx-auto mb-2 text-purple-600" size={24} />
                <p className="text-xs font-medium text-purple-800">Instant Results</p>
              </div>
            </div>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full rounded-lg"
              />
              <canvas ref={canvasRef} className="hidden" />
              
              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center rounded-lg">
                  <div className="text-white text-center">
                    <div className="animate-spin w-8 h-8 border-2 border-white border-t-transparent rounded-full mx-auto mb-2"></div>
                    <p>Analyzing...</p>
                  </div>
                </div>
              )}
              
              <div className="absolute top-4 left-4 right-4 flex justify-between">
                <Badge className="bg-green-500">
                  {isScanning ? 'Scanning...' : 'Ready'}
                </Badge>
                <Button size="sm" onClick={stopCamera} variant="destructive">
                  Stop
                </Button>
              </div>
            </div>
            
            <div className="text-center text-sm text-gray-600">
              <p>Point camera at food label. Analysis will start automatically.</p>
            </div>
          </div>
        )}
        
        {isAnalyzing && !isCameraMode && (
          <div className="text-center py-8">
            <div className="relative">
              <div className="animate-spin w-12 h-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <Brain size={20} className="text-primary animate-pulse" />
              </div>
            </div>
            <p className="text-gray-800 font-medium mb-2">🤖 AI is analyzing your food product...</p>
            <div className="text-sm text-gray-600 space-y-1">
              <p>• Extracting text with OCR technology</p>
              <p>• Identifying ingredients and additives</p>
              <p>• Calculating nutritional score</p>
              <p>• Verifying FSSAI license</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}