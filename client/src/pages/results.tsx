import { useQuery } from '@tanstack/react-query';
import { useRoute } from 'wouter';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { EnhancedAnalysisResults } from '@/components/enhanced-analysis-results';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { AlertCircle, ArrowLeft } from 'lucide-react';
import { Link } from 'wouter';

export default function Results() {
  const [match, params] = useRoute('/results/:id');
  const productId = params?.id;

  const { data: product, isLoading, error } = useQuery({
    queryKey: [`/api/products/${productId}`],
    enabled: !!productId,
  });

  const { data: personalizedData } = useQuery({
    queryKey: [`/api/products/${productId}/personalized`],
    enabled: !!productId,
  });

  const handleFindAlternatives = () => {
    alert('Alternative products feature coming soon!');
  };

  const handleShare = () => {
    if (navigator.share && product) {
      navigator.share({
        title: `${product.productName} - Health Analysis`,
        text: `Check out this food safety analysis from FoodSense AI`,
        url: window.location.href,
      });
    } else {
      // Fallback to copying URL
      navigator.clipboard.writeText(window.location.href);
      alert('Results link copied to clipboard!');
    }
  };

  if (!match) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md mx-4">
          <CardContent className="pt-6 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Invalid Product ID</h1>
            <p className="text-gray-600 mb-6">The product you're looking for doesn't exist.</p>
            <Link href="/scan">
              <Button className="w-full">
                <ArrowLeft className="mr-2" size={16} />
                Back to Scan
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50">
        <TopNavigation />
        <div className="py-16 px-4 flex items-center justify-center">
          <Card className="w-full max-w-md mx-4">
            <CardContent className="pt-6 text-center">
              <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <h1 className="text-2xl font-bold text-gray-900 mb-2">Error Loading Results</h1>
              <p className="text-gray-600 mb-6">
                Failed to load the analysis results. Please try again.
              </p>
              <div className="space-y-3">
                <Link href="/scan">
                  <Button className="w-full">
                    <ArrowLeft className="mr-2" size={16} />
                    Back to Scan
                  </Button>
                </Link>
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={() => window.location.reload()}
                >
                  Retry
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
        <BottomNavigation />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <TopNavigation />
      
      <section className="py-8 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Back Button */}
          <div className="mb-6">
            <Link href="/scan">
              <Button variant="ghost" className="flex items-center space-x-2">
                <ArrowLeft size={16} />
                <span>Back to Scan</span>
              </Button>
            </Link>
          </div>

          {isLoading ? (
            /* Loading Skeletons */
            <div className="space-y-8">
              <Card>
                <div className="bg-gradient-to-r from-primary to-secondary p-6">
                  <Skeleton className="h-8 w-64 bg-white/20 mb-2" />
                  <Skeleton className="h-4 w-32 bg-white/10" />
                </div>
              </Card>
              
              <Card>
                <CardContent className="p-8">
                  <Skeleton className="h-8 w-48 mb-6" />
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="p-6 border rounded-xl">
                        <Skeleton className="h-10 w-10 mx-auto mb-3" />
                        <Skeleton className="h-6 w-24 mx-auto mb-2" />
                        <Skeleton className="h-8 w-16 mx-auto mb-2" />
                        <Skeleton className="h-4 w-20 mx-auto" />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : product ? (
            <EnhancedAnalysisResults
              product={product}
              userAllergies={personalizedData?.allergies}
              userConditions={personalizedData?.healthConditions}
            />
          ) : (
            <Card>
              <CardContent className="pt-6 text-center">
                <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h2 className="text-xl font-semibold text-gray-900 mb-2">No Results Found</h2>
                <p className="text-gray-600">The analysis results are not available.</p>
              </CardContent>
            </Card>
          )}
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}
