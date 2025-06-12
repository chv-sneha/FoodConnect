import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { SafetyBadge } from './safety-badge';
import { getLevelColor } from '@/lib/ingredient-analyzer';
import { AnalysisResult, PersonalizedAlert } from '@/types/analysis';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Search, 
  Share2,
  Clock,
  Beaker,
  Wheat,
  Heart
} from 'lucide-react';

interface AnalysisResultsProps {
  result: AnalysisResult;
  personalizedAlerts?: PersonalizedAlert[];
  onFindAlternatives?: () => void;
  onShare?: () => void;
}

export function AnalysisResults({ 
  result, 
  personalizedAlerts = [], 
  onFindAlternatives,
  onShare 
}: AnalysisResultsProps) {
  const { analysis, safetyScore, productName, fssaiVerified, fssaiNumber } = result;

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Product Header */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-primary to-secondary text-white">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">{productName}</CardTitle>
              <div className="flex items-center space-x-2 text-blue-100 mt-2">
                <Clock size={16} />
                <span>Scanned {result.scannedAt ? new Date(result.scannedAt).toLocaleDateString() : 'just now'}</span>
              </div>
            </div>
            <SafetyBadge score={safetyScore} size="lg" />
          </div>
        </CardHeader>
      </Card>

      {/* Generic Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-3 text-2xl">
            <Beaker className="text-primary" />
            <span>General Health Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Level Indicators */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className={`rounded-xl p-6 text-center border ${getLevelColor(analysis.sugarLevel)}`}>
              <Wheat className="mx-auto mb-3" size={40} />
              <h5 className="font-bold text-lg">Sugar Level</h5>
              <p className="text-3xl font-bold my-2 uppercase">{analysis.sugarLevel}</p>
              <p className="text-sm opacity-75">Based on ingredients</p>
            </div>
            
            <div className={`rounded-xl p-6 text-center border ${getLevelColor(analysis.saltLevel)}`}>
              <div className="w-10 h-10 mx-auto mb-3 bg-current rounded-full opacity-20"></div>
              <h5 className="font-bold text-lg">Salt Level</h5>
              <p className="text-3xl font-bold my-2 uppercase">{analysis.saltLevel}</p>
              <p className="text-sm opacity-75">Sodium content</p>
            </div>
            
            <div className={`rounded-xl p-6 text-center border ${analysis.additiveCount > 3 ? getLevelColor('high') : getLevelColor('low')}`}>
              <AlertTriangle className="mx-auto mb-3" size={40} />
              <h5 className="font-bold text-lg">Additives</h5>
              <p className="text-3xl font-bold my-2">{analysis.additiveCount}</p>
              <p className="text-sm opacity-75">Artificial substances</p>
            </div>
          </div>

          {/* Harmful Ingredients */}
          {analysis.harmfulIngredients.length > 0 && (
            <div className="bg-gray-50 rounded-xl p-6">
              <h5 className="font-bold text-gray-900 mb-4 flex items-center">
                <AlertTriangle className="text-red-500 mr-2" />
                Concerning Ingredients Found
              </h5>
              <div className="space-y-4">
                {analysis.harmfulIngredients.map((ingredient, index) => (
                  <div key={index} className="flex items-center justify-between py-3 border-b border-gray-200 last:border-b-0">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div>
                        <p className="font-semibold text-gray-900">
                          {ingredient.name}
                          {ingredient.commonName && (
                            <span className="text-gray-600 ml-2">({ingredient.commonName})</span>
                          )}
                        </p>
                        <p className="text-sm text-gray-600">{ingredient.description}</p>
                        {ingredient.concerns.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {ingredient.concerns.map((concern, idx) => (
                              <Badge key={idx} variant="destructive" className="text-xs">
                                {concern}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    <Badge variant="destructive">AVOID</Badge>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Personalized Alerts */}
      {personalizedAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-3 text-2xl">
              <Shield className="text-secondary" />
              <span>Your Personal Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {personalizedAlerts.map((alert, index) => (
              <div 
                key={index} 
                className={`
                  p-6 rounded-xl border-l-4
                  ${alert.severity === 'high' ? 'bg-red-50 border-red-500' : 
                    alert.severity === 'medium' ? 'bg-orange-50 border-orange-500' : 
                    'bg-yellow-50 border-yellow-500'}
                `}
              >
                <div className="flex items-center mb-2">
                  {alert.type === 'allergen' ? (
                    <AlertTriangle className="text-red-500 mr-3" />
                  ) : (
                    <Heart className="text-orange-500 mr-3" />
                  )}
                  <h5 className={`font-bold ${
                    alert.severity === 'high' ? 'text-red-900' : 
                    alert.severity === 'medium' ? 'text-orange-900' : 
                    'text-yellow-900'
                  }`}>
                    {alert.title}
                  </h5>
                </div>
                <p className={
                  alert.severity === 'high' ? 'text-red-800' : 
                  alert.severity === 'medium' ? 'text-orange-800' : 
                  'text-yellow-800'
                }>
                  {alert.message}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* FSSAI Verification */}
      <Card>
        <CardContent className="p-6">
          <div className={`rounded-xl p-6 border ${
            fssaiVerified ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Shield className={fssaiVerified ? 'text-green-600' : 'text-red-600'} size={32} />
                <div>
                  <h5 className={`font-bold ${fssaiVerified ? 'text-green-900' : 'text-red-900'}`}>
                    FSSAI Verification
                  </h5>
                  <p className={fssaiVerified ? 'text-green-700' : 'text-red-700'}>
                    {fssaiVerified 
                      ? `License: ${fssaiNumber} - Verified ✓`
                      : 'No valid FSSAI license found'
                    }
                  </p>
                </div>
              </div>
              <Badge 
                variant={fssaiVerified ? "default" : "destructive"}
                className={fssaiVerified ? 'bg-green-100 text-green-800 hover:bg-green-200' : ''}
              >
                {fssaiVerified ? 'LEGAL' : 'UNVERIFIED'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        <Button 
          className="flex-1 bg-primary text-white hover:bg-green-600 py-4 text-lg"
          onClick={onFindAlternatives}
        >
          <Search className="mr-2" />
          Find Healthier Alternatives
        </Button>
        <Button 
          variant="secondary"
          className="flex-1 py-4 text-lg"
          onClick={onShare}
        >
          <Share2 className="mr-2" />
          Share Results
        </Button>
      </div>
    </div>
  );
}
