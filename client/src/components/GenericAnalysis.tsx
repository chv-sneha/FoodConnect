import React, { useState, useRef } from 'react';
import { Camera, Upload, Loader2, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface AnalysisResult {
  success: boolean;
  product_name: string;
  ingredients: string[];
  final_safety_score: number;
  nutri_score: {
    grade: string;
    score: number;
    color: string;
  };
  safety_analysis: {
    overall_score: number;
    ingredients: Array<{
      name: string;
      safety_score: number;
      risk_level: string;
      reason: string;
    }>;
  };
  recommendations: {
    recommendations: string[];
    warnings: string[];
    overall_rating: string;
  };
  fssai: {
    valid: boolean;
    status: string;
    message: string;
  };
}

export default function GenericAnalysis() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch('/api/analyze/generic', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    if (score >= 40) return 'text-orange-600';
    return 'text-red-600';
  };

  const getRiskBg = (score: number) => {
    if (score >= 80) return 'bg-green-100';
    if (score >= 60) return 'bg-yellow-100';
    if (score >= 40) return 'bg-orange-100';
    return 'bg-red-100';
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Generic Food Analysis
        </h1>
        <p className="text-gray-600">
          Scan any food label to get instant safety analysis and health insights
        </p>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          {previewUrl ? (
            <div className="space-y-4">
              <img
                src={previewUrl}
                alt="Selected food label"
                className="max-h-64 mx-auto rounded-lg shadow-md"
              />
              <div className="flex justify-center space-x-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
                >
                  Choose Different Image
                </button>
                <button
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="w-4 h-4" />
                      <span>Analyze Food Label</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-16 h-16 text-gray-400 mx-auto" />
              <div>
                <h3 className="text-lg font-medium text-gray-900">
                  Upload Food Label Image
                </h3>
                <p className="text-gray-500">
                  Take a photo or upload an image of the ingredient label
                </p>
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
              >
                Choose Image
              </button>
            </div>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
          <XCircle className="w-5 h-5 text-red-500" />
          <span className="text-red-700">{error}</span>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Product Info */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Product Information</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-medium text-gray-700">Product Name</h3>
                <p className="text-gray-900">{result.product_name}</p>
              </div>
              <div>
                <h3 className="font-medium text-gray-700">FSSAI Status</h3>
                <p className={`font-medium ${result.fssai.valid ? 'text-green-600' : 'text-red-600'}`}>
                  {result.fssai.status}
                </p>
              </div>
            </div>
          </div>

          {/* Safety Score */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Safety Analysis</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className={`p-4 rounded-lg ${getRiskBg(result.final_safety_score)}`}>
                <h3 className="font-medium text-gray-700">Overall Safety Score</h3>
                <p className={`text-3xl font-bold ${getRiskColor(result.final_safety_score)}`}>
                  {result.final_safety_score}/100
                </p>
              </div>
              <div className={`p-4 rounded-lg bg-${result.nutri_score.color}-100`}>
                <h3 className="font-medium text-gray-700">Nutri-Score</h3>
                <p className="text-3xl font-bold text-gray-900">
                  {result.nutri_score.grade}
                </p>
              </div>
              <div className="p-4 rounded-lg bg-gray-100">
                <h3 className="font-medium text-gray-700">Ingredients Found</h3>
                <p className="text-3xl font-bold text-gray-900">
                  {result.ingredients.length}
                </p>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Health Recommendations</h2>
            <div className="space-y-4">
              {result.recommendations.recommendations.map((rec, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                  <CheckCircle className="w-5 h-5 text-blue-500 mt-0.5" />
                  <span className="text-blue-800">{rec}</span>
                </div>
              ))}
              {result.recommendations.warnings.map((warning, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />
                  <span className="text-yellow-800">{warning}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Ingredient Analysis */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Ingredient Analysis</h2>
            <div className="space-y-3">
              {result.safety_analysis.ingredients.slice(0, 10).map((ingredient, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{ingredient.name}</h4>
                    <p className="text-sm text-gray-600">{ingredient.reason}</p>
                  </div>
                  <div className="text-right">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      ingredient.risk_level === 'Safe' ? 'bg-green-100 text-green-800' :
                      ingredient.risk_level === 'Low' ? 'bg-yellow-100 text-yellow-800' :
                      ingredient.risk_level === 'Medium' ? 'bg-orange-100 text-orange-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      {ingredient.risk_level}
                    </span>
                    <p className="text-sm text-gray-500 mt-1">
                      Score: {ingredient.safety_score}/10
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}