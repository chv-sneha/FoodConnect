import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/queryClient';
import { 
  User, 
  Shield, 
  AlertTriangle, 
  Heart, 
  Save,
  History,
  Settings
} from 'lucide-react';

export default function Profile() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [allergies, setAllergies] = useState<string[]>([]);
  const [healthConditions, setHealthConditions] = useState<string[]>([]);

  // Mock user ID - in real app, this would come from auth context
  const userId = 1;

  const { data: user, isLoading } = useQuery({
    queryKey: [`/api/users/${userId}`],
    onSuccess: (data) => {
      setAllergies(data?.allergies || []);
      setHealthConditions(data?.healthConditions || []);
    }
  });

  const { data: scannedProducts } = useQuery({
    queryKey: [`/api/users/${userId}/products`],
  });

  const updateProfileMutation = useMutation({
    mutationFn: async (data: { allergies: string[]; healthConditions: string[] }) => {
      const response = await apiRequest('PUT', `/api/users/${userId}/profile`, data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/users/${userId}`] });
      toast({
        title: "Profile Updated",
        description: "Your health profile has been saved successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Update Failed",
        description: "Failed to update profile. Please try again.",
        variant: "destructive",
      });
    }
  });

  const allergyOptions = [
    'Nuts', 'Dairy', 'Gluten', 'Soy', 'Eggs', 'Fish', 'Shellfish', 'Sesame'
  ];

  const healthConditionOptions = [
    'Diabetes', 'Hypertension', 'Heart Disease', 'Celiac Disease', 'Lactose Intolerance'
  ];

  const toggleAllergy = (allergy: string) => {
    setAllergies(prev => 
      prev.includes(allergy) 
        ? prev.filter(a => a !== allergy)
        : [...prev, allergy]
    );
  };

  const toggleHealthCondition = (condition: string) => {
    setHealthConditions(prev => 
      prev.includes(condition) 
        ? prev.filter(c => c !== condition)
        : [...prev, condition]
    );
  };

  const handleSaveProfile = () => {
    updateProfileMutation.mutate({ allergies, healthConditions });
  };

  return (
    <div className="min-h-screen bg-gray-50 pb-20">
      <TopNavigation />
      
      <section className="py-8 px-4">
        <div className="max-w-2xl mx-auto space-y-8">
          
          {/* Profile Header */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-2xl">
                <User className="text-primary mr-3" />
                Profile
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-4">
                <div className="w-16 h-16 bg-gradient-to-r from-primary to-secondary rounded-full flex items-center justify-center">
                  <User className="text-white" size={32} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">
                    {user?.username || 'User'}
                  </h3>
                  <p className="text-gray-600">
                    Member since {user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'Recently'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Health Profile Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-xl">
                <Shield className="text-primary mr-3" />
                Health Profile
              </CardTitle>
              <p className="text-gray-600">
                Configure your allergies and health conditions for personalized analysis
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              
              {/* Allergies Section */}
              <div>
                <Label className="text-base font-semibold flex items-center mb-4">
                  <AlertTriangle className="text-orange-500 mr-2" size={20} />
                  Allergies & Intolerances
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  {allergyOptions.map((allergy) => (
                    <label 
                      key={allergy}
                      className="flex items-center space-x-3 p-3 border border-gray-200 rounded-xl hover:border-primary transition-colors cursor-pointer"
                    >
                      <Checkbox 
                        checked={allergies.includes(allergy)}
                        onCheckedChange={() => toggleAllergy(allergy)}
                      />
                      <span className="text-sm font-medium">{allergy}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Health Conditions Section */}
              <div>
                <Label className="text-base font-semibold flex items-center mb-4">
                  <Heart className="text-red-500 mr-2" size={20} />
                  Health Conditions
                </Label>
                <div className="space-y-3">
                  {healthConditionOptions.map((condition) => (
                    <label 
                      key={condition}
                      className="flex items-center space-x-3 p-3 border border-gray-200 rounded-xl hover:border-primary transition-colors cursor-pointer"
                    >
                      <Checkbox 
                        checked={healthConditions.includes(condition)}
                        onCheckedChange={() => toggleHealthCondition(condition)}
                      />
                      <span className="text-sm font-medium">{condition}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Button 
                className="w-full bg-primary text-white hover:bg-green-600 py-3 text-lg"
                onClick={handleSaveProfile}
                disabled={updateProfileMutation.isPending}
              >
                <Save className="mr-2" size={20} />
                {updateProfileMutation.isPending ? 'Saving...' : 'Save Profile'}
              </Button>
            </CardContent>
          </Card>

          {/* Scan History */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-xl">
                <History className="text-secondary mr-3" />
                Scan History
              </CardTitle>
            </CardHeader>
            <CardContent>
              {scannedProducts && scannedProducts.length > 0 ? (
                <div className="space-y-4">
                  {scannedProducts.slice(0, 5).map((product: any) => (
                    <div key={product.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-xl">
                      <div>
                        <h4 className="font-medium text-gray-900">{product.productName}</h4>
                        <p className="text-sm text-gray-600">
                          Scanned {new Date(product.scannedAt).toLocaleDateString()}
                        </p>
                      </div>
                      <div className={`
                        px-3 py-1 rounded-full text-xs font-bold
                        ${product.safetyScore === 'safe' ? 'bg-green-100 text-green-800' :
                          product.safetyScore === 'moderate' ? 'bg-orange-100 text-orange-800' :
                          'bg-red-100 text-red-800'}
                      `}>
                        {product.safetyScore.toUpperCase()}
                      </div>
                    </div>
                  ))}
                  {scannedProducts.length > 5 && (
                    <p className="text-center text-gray-600 pt-4">
                      Showing 5 of {scannedProducts.length} scanned products
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <History className="mx-auto text-gray-400 mb-4" size={48} />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No scans yet</h3>
                  <p className="text-gray-600">Start scanning products to see your history here</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-xl">
                <Settings className="text-gray-600 mr-3" />
                Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button variant="outline" className="w-full justify-start">
                <Settings className="mr-2" size={20} />
                App Preferences
              </Button>
              <Button variant="outline" className="w-full justify-start">
                <Shield className="mr-2" size={20} />
                Privacy Settings
              </Button>
              <Button variant="outline" className="w-full justify-start text-red-600 hover:text-red-700">
                <AlertTriangle className="mr-2" size={20} />
                Delete Account
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}
