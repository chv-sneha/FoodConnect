import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { ChatGPTAnalysisSummary } from './ChatGPTAnalysisSummary';

interface ComprehensiveAnalysisProps {
  analysis: {
    productAndBrand: {
      brandName: string;
      productName: string;
      manufacturer: string;
      netWeight: string;
      shelfLife: string;
    };
    ingredients: {
      list: string[];
      preservatives: string;
      processing: string;
    };
    nutritionFacts: any;
    nutriScoreDistribution: {
      grade: string;
      positivePoints: number;
      negativePoints: number;
      breakdown: {
        positiveFactors: string[];
        negativeFactors: string[];
      };
    };
    ingredientAnalysis: Array<{
      name: string;
      category: string;
      risk: string;
      description: string;
      nutritionalValue: string;
    }>;
    additionalObservations: {
      netWeight: string;
      manufacturer: string;
      fssaiNumbers: string[];
      shelfLife: string;
      packaging: string[];
      authenticity: string[];
    };
  };
}

export const ComprehensiveAnalysisReport: React.FC<ComprehensiveAnalysisProps> = ({ analysis }) => {
  return (
    <div className="max-w-4xl mx-auto p-4">
      {/* Header */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-center flex items-center justify-center gap-2">
            🤖 ChatGPT-Level Food Analysis Report
          </CardTitle>
          <p className="text-center text-gray-600 text-sm">
            Comprehensive ingredient analysis with detailed nutritional breakdown
          </p>
        </CardHeader>
      </Card>

      {/* ChatGPT-Style Analysis */}
      <ChatGPTAnalysisSummary analysis={analysis} />
    </div>
  );
};

export default ComprehensiveAnalysisReport;