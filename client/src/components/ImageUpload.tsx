// Enhanced Image Upload Component for Complete Food Analysis
import React, { useState } from 'react';

interface NutriScore {
  grade: string;
  score: number;
  negativePoints: number;
  positivePoints: number;
}

interface NutritionFacts {
  facts: Record<string, number>;
  healthScore: number;
  safetyLevel: string;
  totalIngredients: number;
  toxicIngredients: number;
  extractedFromText: boolean;
  apiEnhanced: boolean;
}

interface IngredientAnalysis {
  ingredient: string;
  name: string;
  category: string;
  risk: string;
  risk_level: string;
  toxicity_score: number;
  is_toxic: boolean;
  description: string;
  personalRisk?: string[];
}

interface FSSAIResult {
  summary: {
    status: string;
    message: string;
    total_found: number;
    valid_count: number;
  };
}

interface AnalysisResult {
  success: boolean;
  productName: string;
  nutriScore: NutriScore;
  nutrition: NutritionFacts;
  ingredientAnalysis: IngredientAnalysis[];
  fssai: FSSAIResult;
  recommendations: Array<{
    type: string;
    message: string;
    priority?: string;
  }>;
  extractedText: string;
  ocrConfidence: number;
  error?: string;
}

export const ImageUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'generic' | 'customized'>('generic');

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const endpoint = analysisMode === 'generic' ? '/api/generic/analyze' : '/api/customized/analyze';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const data: AnalysisResult = await response.json();
      setResult(data);
      
      if (data.success) {
        console.log('✅ Analysis Success:', data);
      }
      
    } catch (error) {
      setResult({
        success: false,
        productName: '',
        nutriScore: { grade: 'E', score: 0, negativePoints: 0, positivePoints: 0 },
        nutrition: { facts: {}, healthScore: 0, safetyLevel: 'unknown', totalIngredients: 0, toxicIngredients: 0, extractedFromText: false, apiEnhanced: false },
        ingredientAnalysis: [],
        fssai: { summary: { status: 'Error', message: 'Analysis failed', total_found: 0, valid_count: 0 } },
        recommendations: [],
        extractedText: '',
        ocrConfidence: 0,
        error: 'Upload failed'
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'safe': return 'text-green-600';
      case 'low': return 'text-yellow-600';
      case 'medium': return 'text-orange-600';
      case 'high': return 'text-red-600';
      case 'dangerous': return 'text-red-800';
      default: return 'text-gray-600';
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'bg-green-500';
      case 'B': return 'bg-lime-500';
      case 'C': return 'bg-yellow-500';
      case 'D': return 'bg-orange-500';
      case 'E': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-center">🍎 FoodSense AI - Smart Food Analysis</h2>
      
      {/* Analysis Mode Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Analysis Mode:</label>
        <div className="flex gap-4">
          <label className="flex items-center">
            <input
              type="radio"
              value="generic"
              checked={analysisMode === 'generic'}
              onChange={(e) => setAnalysisMode(e.target.value as 'generic')}
              className="mr-2"
            />
            Generic Analysis (No Login Required)
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              value="customized"
              checked={analysisMode === 'customized'}
              onChange={(e) => setAnalysisMode(e.target.value as 'customized')}
              className="mr-2"
            />
            Personalized Analysis
          </label>
        </div>
      </div>

      {/* File Input */}
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">Upload Food Label Image:</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="w-full p-3 border border-gray-300 rounded-lg"
        />
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!selectedFile || isProcessing}
        className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 font-medium"
      >
        {isProcessing ? '🔄 Analyzing Food Label...' : '🚀 Analyze Food Label'}
      </button>

      {/* Results */}
      {result && (
        <div className="mt-8 space-y-6">
          {result.success ? (
            <>
              {/* Product Name & Nutri-Score */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-xl font-bold mb-2">{result.productName}</h3>
                <div className="flex items-center gap-4">
                  <div className={`${getGradeColor(result.nutriScore.grade)} text-white px-4 py-2 rounded-full font-bold text-lg`}>
                    Nutri-Score: {result.nutriScore.grade}
                  </div>
                  <div className="text-lg">
                    Health Score: <span className="font-bold">{result.nutriScore.score}%</span>
                  </div>
                </div>
              </div>

              {/* Nutrition Facts */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-bold text-lg mb-3">📊 Nutrition Analysis</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{result.nutrition.healthScore}%</div>
                    <div className="text-sm">Health Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{result.nutrition.totalIngredients}</div>
                    <div className="text-sm">Total Ingredients</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">{result.nutrition.toxicIngredients}</div>
                    <div className="text-sm">Concerning Ingredients</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold capitalize">{result.nutrition.safetyLevel}</div>
                    <div className="text-sm">Safety Level</div>
                  </div>
                </div>
                
                {/* Nutrition Facts from OCR */}
                {Object.keys(result.nutrition.facts).length > 0 && (
                  <div className="mt-4">
                    <h5 className="font-medium mb-2">Detected Nutrition Facts:</h5>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                      {Object.entries(result.nutrition.facts).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="capitalize">{key}:</span>
                          <span className="font-medium">{value}{key === 'calories' ? '' : 'g'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Ingredient Analysis */}
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h4 className="font-bold text-lg mb-3">🧪 Ingredient Analysis</h4>
                <div className="space-y-2">
                  {result.ingredientAnalysis.slice(0, 10).map((ingredient, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white rounded border">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{ingredient.risk}</span>
                        <span className="font-medium">{ingredient.ingredient}</span>
                        <span className="text-sm text-gray-500">({ingredient.category})</span>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-medium ${getRiskColor(ingredient.risk_level)}`}>
                          {ingredient.toxicity_score}/100
                        </div>
                        {ingredient.personalRisk && ingredient.personalRisk.length > 0 && (
                          <div className="text-xs text-red-600">
                            {ingredient.personalRisk.join(', ')}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* FSSAI Verification */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-bold text-lg mb-3">🛡️ FSSAI Verification</h4>
                <div className="flex items-center gap-2">
                  <span className="text-lg">{result.fssai.summary.status}</span>
                  <span>{result.fssai.summary.message}</span>
                </div>
                {result.fssai.summary.valid_count > 0 && (
                  <div className="text-sm text-green-600 mt-1">
                    ✅ {result.fssai.summary.valid_count} valid license(s) found
                  </div>
                )}
              </div>

              {/* Recommendations */}
              {result.recommendations.length > 0 && (
                <div className="bg-red-50 p-4 rounded-lg">
                  <h4 className="font-bold text-lg mb-3">⚠️ Recommendations</h4>
                  <div className="space-y-2">
                    {result.recommendations.map((rec, index) => (
                      <div key={index} className={`p-2 rounded ${
                        rec.priority === 'critical' ? 'bg-red-100 border-red-300' :
                        rec.priority === 'high' ? 'bg-orange-100 border-orange-300' :
                        'bg-yellow-100 border-yellow-300'
                      } border`}>
                        <div className="font-medium">{rec.type.replace('_', ' ').toUpperCase()}</div>
                        <div className="text-sm">{rec.message}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* OCR Confidence */}
              <div className="text-center text-sm text-gray-500">
                OCR Confidence: {result.ocrConfidence.toFixed(1)}% | 
                Analysis Mode: {analysisMode === 'generic' ? 'Generic' : 'Personalized'}
              </div>
            </>
          ) : (
            <div className="bg-red-50 p-4 rounded-lg">
              <h3 className="font-bold text-red-600 text-lg">❌ Analysis Failed</h3>
              <p className="text-red-700">{result.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};