import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';

interface ChatGPTAnalysisSummaryProps {
  analysis: any;
}

export const ChatGPTAnalysisSummary: React.FC<ChatGPTAnalysisSummaryProps> = ({ analysis }) => {
  if (!analysis) {
    return (
      <Card>
        <CardContent className="text-center py-8">
          <p className="text-gray-500">No comprehensive analysis available</p>
        </CardContent>
      </Card>
    );
  }

  const { productAndBrand, ingredients, nutritionFacts, nutriScoreDistribution, ingredientAnalysis, additionalObservations } = analysis;

  return (
    <div className="space-y-4 text-sm">
      {/* Product & Brand */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">1. Product & Brand Name</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div><strong>Brand Name:</strong> {productAndBrand?.brandName || 'Not specified'}</div>
          <div><strong>Product Name:</strong> {productAndBrand?.productName || 'Not specified'}</div>
          <div><strong>Manufacturer:</strong> {productAndBrand?.manufacturer || 'Not specified'}</div>
          <div><strong>Net Weight:</strong> {productAndBrand?.netWeight || 'Not specified'}</div>
        </CardContent>
      </Card>

      {/* Ingredients */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">2. Ingredients (as listed on packaging)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            {Array.isArray(ingredients?.list) ? ingredients.list.map((ingredient: string, index: number) => (
              <div key={index} className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                <span>{ingredient}</span>
              </div>
            )) : (
              <div className="text-gray-500">No ingredients listed</div>
            )}
          </div>
          <div className="text-xs text-gray-600">
            <strong>Preservatives:</strong> {ingredients?.preservatives || 'Not specified'}
          </div>
        </CardContent>
      </Card>

      {/* Nutrition Facts */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">3. Nutrition Facts (per 100g)</CardTitle>
        </CardHeader>
        <CardContent>
          {typeof nutritionFacts === 'object' && nutritionFacts !== null ? (
            <div className="grid grid-cols-2 gap-2 text-sm">
              {Object.entries(nutritionFacts).map(([key, value]) => (
                <div key={key} className="flex justify-between">
                  <span className="capitalize">{key.replace(/([A-Z])/g, ' $1')}:</span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">{String(nutritionFacts || 'No nutrition information available')}</p>
          )}
        </CardContent>
      </Card>

      {/* Nutri-Score */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">4. Nutri-Score Distribution</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-center">
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full text-xl font-bold ${
              nutriScoreDistribution?.grade === 'A' ? 'bg-green-500 text-white' :
              nutriScoreDistribution?.grade === 'B' ? 'bg-lime-500 text-white' :
              nutriScoreDistribution?.grade === 'C' ? 'bg-yellow-500 text-white' :
              nutriScoreDistribution?.grade === 'D' ? 'bg-orange-500 text-white' :
              'bg-red-500 text-white'
            }`}>
              {nutriScoreDistribution?.grade || 'C'}
            </div>
            <p className="mt-1 text-xs text-gray-600">Estimated Nutri-Score</p>
          </div>
          
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <h5 className="font-medium text-green-700 mb-1">Positive Points:</h5>
              <ul className="space-y-0.5">
                {Array.isArray(nutriScoreDistribution?.breakdown?.positiveFactors) ? 
                  nutriScoreDistribution.breakdown.positiveFactors.map((factor: string, index: number) => (
                    <li key={index} className="flex items-center gap-1">
                      <span className="w-1 h-1 bg-green-500 rounded-full"></span>
                      {factor}
                    </li>
                  )) : (
                    <li className="text-gray-500 text-xs">No positive factors identified</li>
                  )
                }
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-red-700 mb-1">Negative Points:</h5>
              <ul className="space-y-0.5">
                {Array.isArray(nutriScoreDistribution?.breakdown?.negativeFactors) ? 
                  nutriScoreDistribution.breakdown.negativeFactors.map((factor: string, index: number) => (
                    <li key={index} className="flex items-center gap-1">
                      <span className="w-1 h-1 bg-red-500 rounded-full"></span>
                      {factor}
                    </li>
                  )) : (
                    <li className="text-gray-500 text-xs">No negative factors identified</li>
                  )
                }
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Ingredient Analysis */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">5. Ingredient Analysis</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {Array.isArray(ingredientAnalysis) ? ingredientAnalysis.map((ingredient: any, index: number) => (
            <div key={index} className="border rounded-lg p-3">
              <div className="flex items-center justify-between mb-1">
                <h5 className="font-medium text-sm">{ingredient.name || 'Unknown ingredient'}</h5>
                <Badge variant={ingredient.risk === '🟢' ? 'default' : ingredient.risk === '🟡' ? 'secondary' : 'destructive'} className="text-xs">
                  {ingredient.category || 'Unknown'}
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">{ingredient.description || 'No description available'}</p>
              <p className="text-xs font-medium text-blue-700">{ingredient.nutritionalValue || 'No nutritional information'}</p>
            </div>
          )) : (
            <div className="text-gray-500 text-center py-4">No ingredient analysis available</div>
          )}
        </CardContent>
      </Card>

      {/* Additional Observations */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">6. Additional Observations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div><strong>Net Weight:</strong> {additionalObservations?.netWeight}</div>
          <div><strong>Manufactured by:</strong> {additionalObservations?.manufacturer}</div>
          {Array.isArray(additionalObservations?.fssaiNumbers) && additionalObservations.fssaiNumbers.length > 0 && (
            <div>
              <strong>FSSAI License Numbers:</strong>
              <div className="ml-4 mt-1">
                {additionalObservations.fssaiNumbers.map((number: string, index: number) => (
                  <div key={index}>• {number}</div>
                ))}
              </div>
            </div>
          )}
          <div><strong>Shelf Life:</strong> {additionalObservations?.shelfLife}</div>
          <div>
            <strong>Packaging Design:</strong> {Array.isArray(additionalObservations?.packaging) ? additionalObservations.packaging.join(', ') : 'Standard packaging'}
          </div>
          <div>
            <strong>Authenticity:</strong> {Array.isArray(additionalObservations?.authenticity) ? additionalObservations.authenticity.join(', ') : 'Standard markings'}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};