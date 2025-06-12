import { useState } from 'react';
import { useLocation } from 'wouter';
import { useMutation } from '@tanstack/react-query';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { UploadZone } from '@/components/upload-zone';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { apiRequest } from '@/lib/queryClient';
import { Shield, AlertTriangle, Heart, Wheat } from 'lucide-react';

export default function Scan() {
  const [, setLocation] = useLocation();
  const [selectedAllergies, setSelectedAllergies] = useState<string[]>([]);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [uploadData, setUploadData] = useState<{
    file: File;
    extractedText: string;
    productName: string;
  } | null>(null);

  const allergies = [
    'Nuts', 'Dairy', 'Gluten', 'Soy', 'Eggs', 'Fish', 'Shellfish', 'Sesame'
  ];

  const healthConditions = [
    'Diabetes', 'Hypertension', 'Heart Disease', 'Celiac Disease', 'Lactose Intolerance'
  ];

  const analyzeProductMutation = useMutation({
    mutationFn: async (data: FormData) => {
      const response = await apiRequest('POST', '/api/products/analyze', data);
      return response.json();
    },
    onSuccess: (result) => {
      // Redirect to results page with the product ID
      setLocation(`/results/${result.id}`);
    },
    onError: (error) => {
      console.error('Analysis failed:', error);
      alert('Failed to analyze product. Please try again.');
    }
  });

  const handleImageAnalyzed = (result: {
    file: File;
    extractedText: string;
    productName: string;
  }) => {
    setUploadData(result);
  };

  const handleAnalyze = () => {
    if (!uploadData) return;

    const formData = new FormData();
    formData.append('image', uploadData.file);
    formData.append('extractedText', uploadData.extractedText);
    formData.append('productName', uploadData.productName);
    formData.append('userId', '1'); // Mock user ID for demo

    analyzeProductMutation.mutate(formData);
  };

  const toggleAllergy = (allergy: string) => {
    setSelectedAllergies(prev => 
      prev.includes(allergy) 
        ? prev.filter(a => a !== allergy)
        : [...prev, allergy]
    );
  };

  const toggleCondition = (condition: string) => {
    setSelectedConditions(prev => 
      prev.includes(condition) 
        ? prev.filter(c => c !== condition)
        : [...prev, condition]
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <TopNavigation />
      
      <section className="py-16 px-4">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Scan Your Food Product
            </h2>
            <p className="text-xl text-gray-600">
              Take a photo or upload the back side of any packaged food
            </p>
          </div>

          <div className="mb-8">
            <UploadZone 
              onImageAnalyzed={handleImageAnalyzed}
              isLoading={analyzeProductMutation.isPending}
            />
          </div>

          {/* Quick Health Profile */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-2xl">
                <Shield className="text-primary mr-3" />
                Quick Health Profile
              </CardTitle>
              <p className="text-gray-600">
                Set your dietary restrictions and health conditions for personalized analysis
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Allergies Section */}
              <div>
                <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                  <AlertTriangle className="text-orange-500 mr-2" size={20} />
                  Allergies & Intolerances
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {allergies.map((allergy) => (
                    <label 
                      key={allergy}
                      className="flex items-center space-x-3 p-3 border border-gray-200 rounded-xl hover:border-primary transition-colors cursor-pointer"
                    >
                      <Checkbox 
                        checked={selectedAllergies.includes(allergy)}
                        onCheckedChange={() => toggleAllergy(allergy)}
                      />
                      <span className="text-sm font-medium">{allergy}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Health Conditions Section */}
              <div>
                <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                  <Heart className="text-red-500 mr-2" size={20} />
                  Health Conditions
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {healthConditions.map((condition) => (
                    <label 
                      key={condition}
                      className="flex items-center space-x-3 p-3 border border-gray-200 rounded-xl hover:border-primary transition-colors cursor-pointer"
                    >
                      <Checkbox 
                        checked={selectedConditions.includes(condition)}
                        onCheckedChange={() => toggleCondition(condition)}
                      />
                      <span className="text-sm font-medium">{condition}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Analyze Button */}
              <Button 
                className="w-full bg-secondary text-white hover:bg-blue-600 py-4 text-lg font-semibold"
                onClick={handleAnalyze}
                disabled={!uploadData || analyzeProductMutation.isPending}
              >
                {analyzeProductMutation.isPending ? 'Analyzing...' : 'Analyze Product'}
              </Button>

              {uploadData && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl">
                  <div className="flex items-center space-x-2 text-green-800">
                    <Wheat size={20} />
                    <span className="font-medium">Ready to analyze: {uploadData.productName}</span>
                  </div>
                  <p className="text-sm text-green-600 mt-1">
                    Image processed successfully. Click "Analyze Product" to continue.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}
