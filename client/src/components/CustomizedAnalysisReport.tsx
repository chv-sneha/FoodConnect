import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Scale, 
  ShoppingCart,
  Download,
  Eye
} from 'lucide-react';

interface AnalysisResult {
  foodItem: string;
  allergenWarnings: Array<{
    ingredient: string;
    severity: 'high' | 'medium' | 'low';
    message: string;
  }>;
  healthMatches: Array<{
    aspect: string;
    status: 'good' | 'moderate' | 'bad';
    message: string;
  }>;
  ageAppropriate: boolean;
  activityMatch: string;
  recommendation: 'avoid' | 'moderate' | 'good';
  alternatives: string[];
  consumptionTime: string;
  portionSuggestion: string;
}

interface CustomizedAnalysisReportProps {
  result: AnalysisResult;
  onSaveReport?: () => void;
  onViewAlternatives?: () => void;
}

export default function CustomizedAnalysisReport({ 
  result, 
  onSaveReport, 
  onViewAlternatives 
}: CustomizedAnalysisReportProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good': return <CheckCircle className="text-green-600" size={16} />;
      case 'moderate': return <AlertTriangle className="text-orange-500" size={16} />;
      case 'bad': return <XCircle className="text-red-600" size={16} />;
      default: return <AlertTriangle className="text-gray-500" size={16} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-800 bg-green-50';
      case 'moderate': return 'text-orange-800 bg-orange-50';
      case 'bad': return 'text-red-800 bg-red-50';
      default: return 'text-gray-800 bg-gray-50';
    }
  };

  const getSeverityEmoji = (severity: string) => {
    switch (severity) {
      case 'high': return '🔴';
      case 'medium': return '🟠';
      case 'low': return '🟡';
      default: return '⚪';
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto shadow-lg">
      <CardHeader className="bg-gradient-to-r from-primary/5 to-secondary/5">
        <CardTitle className="text-2xl font-bold text-center">
          ✅ Customized Food Analysis Report
        </CardTitle>
        <div className="text-center">
          <Badge variant="outline" className="text-lg px-4 py-2">
            🟢 Food Item: {result.foodItem}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-6 space-y-6">
        {/* Allergen Warnings */}
        {result.allergenWarnings.length > 0 && (
          <div className="space-y-3">
            <h3 className="font-semibold text-lg">⚠️ Allergen Alerts</h3>
            {result.allergenWarnings.map((warning, index) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                warning.severity === 'high' ? 'border-red-500 bg-red-50' :
                warning.severity === 'medium' ? 'border-orange-500 bg-orange-50' :
                'border-yellow-500 bg-yellow-50'
              }`}>
                <div className="flex items-center space-x-2">
                  <span>{getSeverityEmoji(warning.severity)}</span>
                  <span className="font-medium">{warning.message}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Health Matches */}
        <div className="space-y-3">
          <h3 className="font-semibold text-lg">🎯 Health Analysis</h3>
          {result.healthMatches.map((match, index) => (
            <div key={index} className={`p-3 rounded-lg flex items-center space-x-3 ${getStatusColor(match.status)}`}>
              {getStatusIcon(match.status)}
              <span>{match.message}</span>
            </div>
          ))}
        </div>

        {/* Age & Activity */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className={`p-3 rounded-lg ${result.ageAppropriate ? 'bg-blue-50 text-blue-800' : 'bg-red-50 text-red-800'}`}>
            <div className="flex items-center space-x-2">
              <span>🔵</span>
              <span>{result.ageAppropriate ? 'Age appropriate' : 'Not suitable for your age'}</span>
            </div>
          </div>
          <div className="p-3 rounded-lg bg-green-50 text-green-800">
            <div className="flex items-center space-x-2">
              <span>✅</span>
              <span>{result.activityMatch}</span>
            </div>
          </div>
        </div>

        {/* Recommendation */}
        <div className={`p-4 rounded-lg border-2 ${
          result.recommendation === 'avoid' ? 'border-red-500 bg-red-50' :
          result.recommendation === 'moderate' ? 'border-orange-500 bg-orange-50' :
          'border-green-500 bg-green-50'
        }`}>
          <div className="text-center">
            <h3 className="font-bold text-lg mb-2">
              {result.recommendation === 'avoid' ? '🚫 Recommendation: Avoid this product' :
               result.recommendation === 'moderate' ? '⚠️ Recommendation: Consume with caution' :
               '✅ Recommendation: Safe to consume'}
            </h3>
            {result.alternatives.length > 0 && (
              <p className="text-sm">
                ✅ Alternative: {result.alternatives.join(', ')}
              </p>
            )}
          </div>
        </div>

        {/* Consumption Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-blue-50 text-blue-800">
            <div className="flex items-center space-x-2">
              <Clock size={16} />
              <span>🍽 Recommended Time: {result.consumptionTime}</span>
            </div>
          </div>
          <div className="p-3 rounded-lg bg-purple-50 text-purple-800">
            <div className="flex items-center space-x-2">
              <Scale size={16} />
              <span>🔁 Portion: {result.portionSuggestion}</span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 pt-4">
          <Button 
            onClick={onSaveReport}
            className="flex items-center space-x-2 bg-primary hover:bg-primary/90"
          >
            <Download size={16} />
            <span>Save Report / Download PDF</span>
          </Button>
          <Button 
            onClick={onViewAlternatives}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <Eye size={16} />
            <span>View Similar Products</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}