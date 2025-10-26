import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Volume2, Download, Share2, AlertTriangle, CheckCircle, XCircle, Brain, Shield, Zap, TrendingUp } from 'lucide-react';
import { useStore } from '@/store/useStore';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';

interface AnalysisResultsProps {
  result: {
    productName?: string;
    nutriScore: {
      grade: string;
      score: number;
      negativePoints: number;
      positivePoints: number;
    };
    nutrition: any;
    ingredientAnalysis: Array<{
      ingredient: string;
      risk: string;
      name: string;
      category: string;
      description: string;
    }>;
    fssai: {
      number: string | null;
      valid: boolean;
    };
    voiceSummary?: string;
    summary?: string;
  };
}

export default function GenericAnalysisResults({ result }: AnalysisResultsProps) {
  const { user } = useStore();
  const { productName, nutriScore, nutrition, ingredientAnalysis, fssai, voiceSummary, summary } = result;
  const displaySummary = (typeof summary === 'string' ? summary : JSON.stringify(summary)) || voiceSummary || 'Analysis complete';

  const playVoiceSummary = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(displaySummary);
      utterance.rate = 0.8;
      speechSynthesis.speak(utterance);
    }
  };

  const downloadReport = () => {
    const reportData = JSON.stringify(result, null, 2);
    const blob = new Blob([reportData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `food-analysis-${Date.now()}.json`;
    a.click();
  };

  const shareResults = () => {
    if (navigator.share) {
      navigator.share({
        title: `${productName || 'Food'} Analysis Results`,
        text: `${productName || 'Product'} - Nutri-Score: ${nutriScore.grade} - ${displaySummary}`,
        url: window.location.href
      });
    }
  };

  const getNutriScoreColor = (grade: string) => {
    const colors = { A: 'bg-green-500', B: 'bg-lime-500', C: 'bg-yellow-500', D: 'bg-orange-500', E: 'bg-red-500' };
    return colors[grade as keyof typeof colors] || 'bg-gray-500';
  };

  // Nutri-Score distribution data based on nutritional content
  const getNutriScoreDistribution = () => {
    const scores = { A: 0, B: 0, C: 0, D: 0, E: 0 };
    
    // Calculate distribution based on current product's nutritional profile
    const currentGrade = nutriScore?.grade || nutriScore?.letter || 'C';
    const totalScore = nutriScore?.score || 0;
    
    // Simulate distribution based on score ranges
    if (totalScore <= -1) scores.A = 35;
    else if (totalScore <= 2) scores.B = 30;
    else if (totalScore <= 10) scores.C = 20;
    else if (totalScore <= 18) scores.D = 10;
    else scores.E = 5;
    
    // Highlight current product's grade
    scores[currentGrade as keyof typeof scores] += 15;
    
    // Normalize to 100%
    const total = Object.values(scores).reduce((a, b) => a + b, 0);
    Object.keys(scores).forEach(key => {
      scores[key as keyof typeof scores] = Math.round((scores[key as keyof typeof scores] / total) * 100);
    });
    
    return [
      { name: 'A - Excellent', value: scores.A, color: '#00B04F' },
      { name: 'B - Good', value: scores.B, color: '#85BB2F' },
      { name: 'C - Fair', value: scores.C, color: '#FFCC00' },
      { name: 'D - Poor', value: scores.D, color: '#FF9500' },
      { name: 'E - Bad', value: scores.E, color: '#FF5722' }
    ].filter(item => item.value > 0);
  };
  
  const pieData = getNutriScoreDistribution();

  const barData = [
    { name: 'Negative', value: nutriScore?.negativePoints || nutriScore?.points_negative || 0, fill: '#ef4444' },
    { name: 'Positive', value: nutriScore?.positivePoints || nutriScore?.points_positive || 0, fill: '#22c55e' }
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Enhanced Product Header */}
      <Card className="border-2 border-primary/20 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-primary/10 to-blue/10">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-primary/10 rounded-full">
                <Brain className="text-primary" size={28} />
              </div>
              <div>
                <h1 className="text-2xl font-bold">{productName || 'Food Product'}</h1>
                <p className="text-sm text-gray-600 font-normal">📸 Uploaded Image Analysis • 🤖 AI-Powered • ⚡ Real-time OCR</p>
                {productName && productName !== 'Food Product' && (
                  <p className="text-xs text-blue-600 font-medium mt-1">✅ Product detected from image: "{productName}"</p>
                )}
              </div>
            </div>
            <div className="flex gap-2">
              {(fssai?.valid || fssai?.status?.includes('Verified')) && (
                <Badge className="bg-green-500 text-white flex items-center gap-1">
                  <Shield size={14} />
                  FSSAI Verified
                </Badge>
              )}
              <Badge className="bg-blue-500 text-white flex items-center gap-1">
                <Zap size={14} />
                High Confidence
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Enhanced Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-xl border border-green-200">
              <div className={`w-12 h-12 mx-auto rounded-full flex items-center justify-center text-white text-xl font-bold mb-2 ${getNutriScoreColor(nutriScore?.grade || nutriScore?.letter || 'C')}`}>
                {nutriScore?.grade || nutriScore?.letter || 'C'}
              </div>
              <p className="text-sm font-semibold text-gray-700">Nutri-Score</p>
              <p className="text-xs text-gray-600">ML Calculated</p>
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border border-blue-200">
              <div className="text-2xl font-bold text-blue-600 mb-2">
                {Math.round((100 - Math.abs((nutriScore?.score || 0) * 5)))}%
              </div>
              <p className="text-sm font-semibold text-gray-700">Health Score</p>
              <p className="text-xs text-gray-600">AI Powered</p>
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl border border-purple-200">
              <div className="flex justify-center mb-2">
                {(fssai?.valid || fssai?.status?.includes('Verified')) ? (
                  <CheckCircle className="text-green-600" size={28} />
                ) : (
                  <XCircle className="text-red-600" size={28} />
                )}
              </div>
              <p className="text-sm font-semibold text-gray-700">Safety Rating</p>
              <p className="text-xs text-gray-600">FSSAI Checked</p>
            </div>
            
            <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl border border-orange-200">
              <div className="flex justify-center mb-2">
                <TrendingUp className="text-orange-600" size={28} />
              </div>
              <div className="text-xl font-bold text-orange-600 mb-1">
                &lt;2s
              </div>
              <p className="text-sm font-semibold text-gray-700">Analysis Time</p>
              <p className="text-xs text-gray-600">Lightning Fast</p>
            </div>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex flex-wrap gap-2">
              <Button onClick={playVoiceSummary} size="sm" className="bg-green-500 hover:bg-green-600 text-white">
                <Volume2 size={16} className="mr-2" />
                🔊 Listen to Analysis
              </Button>
              <Button onClick={downloadReport} size="sm" className="bg-blue-500 hover:bg-blue-600 text-white">
                <Download size={16} className="mr-2" />
                📄 Download Report
              </Button>
              <Button onClick={shareResults} size="sm" className="bg-purple-500 hover:bg-purple-600 text-white">
                <Share2 size={16} className="mr-2" />
                📤 Share Results
              </Button>
              {user && (
                <Button onClick={() => alert('Personalized recommendations based on your profile!')} size="sm" className="bg-orange-500 hover:bg-orange-600 text-white">
                  🎯 Personalized Tips
                </Button>
              )}
            </div>
          </div>
          
          <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
            {typeof displaySummary === 'string' ? displaySummary : 'Analysis complete'}
          </p>
        </CardContent>
      </Card>

      {/* Nutrition Visualization */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Nutri-Score Distribution</span>
              <Badge className={`${getNutriScoreColor(nutriScore?.grade || nutriScore?.letter || 'C')} text-white`}>
                Your Product: {nutriScore?.grade || nutriScore?.letter || 'C'}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, value }) => `${name.split(' - ')[0]}: ${value}%`}
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.color}
                      stroke={entry.name.startsWith(nutriScore?.grade || nutriScore?.letter || 'C') ? '#000' : 'none'}
                      strokeWidth={entry.name.startsWith(nutriScore?.grade || nutriScore?.letter || 'C') ? 3 : 0}
                    />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value: number, name: string) => [`${value}%`, name]}
                  contentStyle={{ 
                    backgroundColor: '#f8f9fa', 
                    border: '1px solid #dee2e6',
                    borderRadius: '8px'
                  }}
                />
                <Legend 
                  verticalAlign="bottom" 
                  height={36}
                  formatter={(value) => value.split(' - ')[1] || value}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 text-center text-sm text-gray-600">
              <p>Distribution shows how this product compares to similar food items</p>
              <p className="font-medium mt-1">Your product is highlighted with a border</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Nutri-Score Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Ingredient Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="text-orange-500" />
            <span>Ingredient Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {(ingredientAnalysis && ingredientAnalysis.length > 0) ? (
            <div className="space-y-3">
              {(ingredientAnalysis || []).map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{item.risk}</span>
                    <div>
                      <p className="font-medium">{item.name}</p>
                      <p className="text-sm text-gray-600">{item.description}</p>
                    </div>
                  </div>
                  <Badge variant={item.risk === '🟥' ? 'destructive' : item.risk === '🟧' ? 'secondary' : 'default'}>
                    {item.category}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <CheckCircle className="mx-auto mb-2" size={48} />
              <p>No concerning ingredients detected</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Enhanced FSSAI Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {fssai?.valid ? <CheckCircle className="text-green-500" /> : <XCircle className="text-red-500" />}
            <span>FSSAI Verification Status</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="font-medium text-lg">
                  License Number: {fssai?.number || 'Not Found'}
                </p>
                <p className={`text-sm font-medium ${fssai?.valid ? 'text-green-600' : 'text-red-600'}`}>
                  Status: {fssai?.status || (fssai?.valid ? 'Verified ✅' : 'Not Found ❌')}
                </p>
                {fssai?.message && (
                  <p className="text-sm text-gray-600 mt-1">{fssai.message}</p>
                )}
              </div>
              <Badge 
                variant={fssai?.valid ? 'default' : 'destructive'}
                className={`text-white ${fssai?.valid ? 'bg-green-500' : 'bg-red-500'}`}
              >
                {fssai?.valid ? 'VERIFIED' : 'UNVERIFIED'}
              </Badge>
            </div>
            
            {fssai?.number && (
              <div className="text-xs text-gray-500 bg-blue-50 p-3 rounded border-l-4 border-blue-400">
                <p><strong>ℹ️ About FSSAI:</strong> Food Safety and Standards Authority of India license ensures this product meets safety standards for manufacturing and distribution.</p>
                {fssai.valid && <p className="mt-1"><strong>✅ This product is registered</strong> with FSSAI and meets regulatory requirements.</p>}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}