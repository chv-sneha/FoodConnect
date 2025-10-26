import { useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { useAuth } from '@/context/AuthContext';
import { User, Save, ArrowLeft } from 'lucide-react';
import { Link } from 'wouter';

export default function ProfileEdit() {
  const { user } = useAuth();
  const [, setLocation] = useLocation();
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    allergies: user?.allergies || [],
    healthConditions: user?.healthConditions || [],
    age: user?.age || '',
    activityLevel: user?.activityLevel || '',
    healthGoal: user?.healthGoal || ''
  });

  const allergies = [
    'Nuts', 'Dairy', 'Gluten', 'Soy', 'Eggs', 'Fish', 'Shellfish', 'Sesame'
  ];

  const healthConditions = [
    'Diabetes', 'Hypertension', 'Heart Disease', 'Celiac Disease', 'Lactose Intolerance'
  ];

  const handleSave = async () => {
    // Update user profile logic
    console.log('Saving profile:', formData);
    setLocation('/profile');
  };

  const toggleAllergy = (allergy: string) => {
    setFormData(prev => ({
      ...prev,
      allergies: prev.allergies.includes(allergy)
        ? prev.allergies.filter(a => a !== allergy)
        : [...prev.allergies, allergy]
    }));
  };

  const toggleCondition = (condition: string) => {
    setFormData(prev => ({
      ...prev,
      healthConditions: prev.healthConditions.includes(condition)
        ? prev.healthConditions.filter(c => c !== condition)
        : [...prev.healthConditions, condition]
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <TopNavigation />
      
      <section className="py-8 px-4">
        <div className="max-w-2xl mx-auto">
          <div className="mb-6">
            <Link href="/profile">
              <Button variant="ghost" className="flex items-center space-x-2">
                <ArrowLeft size={16} />
                <span>Back to Profile</span>
              </Button>
            </Link>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-2xl">
                <User className="mr-3" />
                Edit Profile
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="name">Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div>
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                  />
                </div>
              </div>

              {/* Allergies */}
              <div>
                <Label className="text-base font-semibold mb-4 block">Allergies</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {allergies.map((allergy) => (
                    <label key={allergy} className="flex items-center space-x-2 cursor-pointer">
                      <Checkbox
                        checked={formData.allergies.includes(allergy)}
                        onCheckedChange={() => toggleAllergy(allergy)}
                      />
                      <span className="text-sm">{allergy}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Health Conditions */}
              <div>
                <Label className="text-base font-semibold mb-4 block">Health Conditions</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {healthConditions.map((condition) => (
                    <label key={condition} className="flex items-center space-x-2 cursor-pointer">
                      <Checkbox
                        checked={formData.healthConditions.includes(condition)}
                        onCheckedChange={() => toggleCondition(condition)}
                      />
                      <span className="text-sm">{condition}</span>
                    </label>
                  ))}
                </div>
              </div>

              <Button onClick={handleSave} className="w-full">
                <Save className="mr-2" size={16} />
                Save Changes
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}