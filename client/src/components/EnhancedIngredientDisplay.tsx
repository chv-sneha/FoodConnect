import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, CheckCircle, XCircle, Info, TrendingUp, Shield } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

interface EnhancedIngredientDisplayProps {
  result: {
    ingredients: Array<{
      id: string;
      name: string;
      scientificName?: string;
      commonNames: string[];
      category: string;
      safetyProfile: {
        level: 'safe' | 'moderate' | 'caution' | 'avoid';
        dailyLimit?: number;
        maxUsageLevel?: number;
        regulatoryStatus: string[];
        warnings: string[];
      };
      quantityInfo: {
        typicalRange: [number, number];
        measurementUnit: string;
        servingSizeContext?: string;
      };
      healthImpact: {
        benefits?: string[];
        concerns?: string[];
        contraindications?: string[];
        interactions?: string[];
      };
      source: 'natural' | 'synthetic' | 'processed';
      vegan: boolean;
      allergens: string[];
      certifications: string[];
      safetyAssessment: any;
      userFriendlyInfo: any;
    }>;
    overallSafety: any;
    summary: any;
    recommendations: string[];
  };
}

export function EnhancedIngredientDisplay({ result }: EnhancedIngredientDisplayProps) {
  const { ingredients, overallSafety, recommendations } = result;

  const getSafetyColor = (level: string) => {
    const colors = {
      safe: 'bg-green-500',
      moderate: 'bg-yellow-500',
      caution: 'bg-orange-500',
      avoid: 'bg-red-500'
    };
    return colors[level] || 'bg-gray-500';
  };

  const getSafetyIcon = (level: string) => {
    const icons = {
      safe: <CheckCircle className="w-4 h-4" />,
      moderate: <AlertTriangle className="w-4 h-4" />,
      caution: <AlertTriangle className="w-4 h-4" />,
      avoid: <XCircle className="w-4 h-4" />
    };
    return icons[level] || <Info className="w-4 h-4" />;
  };

  return (
    <div className="space-y-6">
      {/* Overall Safety Summary */}
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="w-5 h-5" />
            <span>Ingredient Safety Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center text-white ${getSafetyColor(overallSafety.level)}`}>
                {getSafetyIcon(overallSafety.level)}
              </div>
              <p className="mt-2 font-semibold">Overall Safety</p>
              <p className="text-sm text-gray-600 capitalize">{overallSafety.level}</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 mx-auto rounded-full bg-blue-500 flex items-center justify-center text-white">
                <span className="text-xl font-bold">{ingredients.length}</span>
              </div>
              <p className="mt-2 font-semibold">Ingredients</p>
              <p className="text-sm text-gray-600">Analyzed</p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 mx-auto rounded-full bg-purple-500 flex items-center justify-center text-white">
                <TrendingUp className="w-6 h-6" />
              </div>
              <p className="mt-2 font-semibold">Health Score</p>
              <p className="text-sm text-gray-600">Calculated</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Ingredient Cards */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Detailed Ingredient Analysis</h3>
        {ingredients.map((ingredient, index) => (
          <IngredientCard key={index} ingredient={ingredient} index={index} />
        ))}
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recommendations</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {recommendations.map((rec, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span className="text-sm">{rec}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function IngredientCard({ ingredient, index }: { ingredient: any; index: number }) {
  const getSafetyColor = (level: string) => {
    const colors = {
      safe: 'text-green-600 bg-green-50',
      moderate: 'text-yellow-600 bg-yellow-50',
      caution: 'text-orange-600 bg-orange-50',
      avoid: 'text-red-600 bg-red-50'
    };
    return colors[level] || 'text-gray-600 bg-gray-50';
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{ingredient.name}</CardTitle>
            {ingredient.scientificName && (
              <p className="text-sm text-gray-600 italic">{ingredient.scientificName}</p>
            )}
          </div>
          <Badge className={`${getSafetyColor(ingredient.safetyProfile.level)}`}>
            {ingredient.safetyProfile.level.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Basic Info */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium">Category:</span> {ingredient.category}
          </div>
          <div>
            <span className="font-medium">Source:</span> {ingredient.source}
          </div>
          <div>
            <span className="font-medium">Vegan:</span> {ingredient.vegan ? '✅' : '❌'}
          </div>
          <div>
            <span className="font-medium">Allergens:</span> {ingredient.allergens.join(', ') || 'None'}
          </div>
        </div>

        {/* Quantity Info */}
        <div>
          <h4 className="font-medium mb-2">Quantity Information</h4>
          <div className="text-sm text-gray-600">
            <p>Typical range: {ingredient.quantityInfo.typicalRange[0]}-{ingredient.quantityInfo.typicalRange[1]}%</p>
            <p>Context: {ingredient.quantityInfo.servingSizeContext}</p>
          </div>
        </div>

        {/* Health Impact */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {ingredient.healthImpact.benefits && ingredient.healthImpact.benefits.length > 0 && (
            <div>
              <h4 className="font-medium text-green-600 mb-2">Benefits</h4>
              <ul className="text-sm space-y-1">
                {ingredient.healthImpact.benefits.map((benefit, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    {benefit}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {ingredient.healthImpact.concerns && ingredient.healthImpact.concerns.length > 0 && (
            <div>
              <h4 className="font-medium text-orange-600 mb-2">Concerns</h4>
              <ul className="text-sm space-y-1">
                {ingredient.healthImpact.concerns.map((concern, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-orange-500 mr-2">•</span>
                    {concern}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Certifications */}
        {ingredient.certifications.length > 0 && (
          <div>
            <h4 className="font-medium mb-2">Certifications</h4>
            <div className="flex flex-wrap gap-2">
              {ingredient.certifications.map((cert, i) => (
                <Badge key={i} variant="outline" className="text-xs">
                  {cert}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Regulatory Status */}
        <div>
          <h4 className="font-medium mb-2">Regulatory Status</h4>
          <div className="flex flex-wrap gap-2">
            {ingredient.safetyProfile.regulatoryStatus.map((status, i) => (
              <Badge key={i} variant="secondary" className="text-xs">
                {status}
              </Badge>
            ))}
          </div>
        </div>

        {/* Warnings */}
        {ingredient.safetyProfile.warnings.length > 0 && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
            <h4 className="font-medium text-yellow-800 mb-2">⚠️ Important Notes</h4>
            <ul className="text-sm text-yellow-700 space-y-1">
              {ingredient.safetyProfile.warnings.map((warning, i) => (
                <li key={i}>• {warning}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
