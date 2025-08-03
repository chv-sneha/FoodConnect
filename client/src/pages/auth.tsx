import { useState } from 'react';
import { useLocation } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useAuth } from '@/context/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { 
  User, 
  Mail, 
  Lock, 
  AlertTriangle, 
  Heart, 
  Leaf,
  ArrowLeft,
  Plus,
  X
} from 'lucide-react';
import { Link } from 'wouter';

export default function Auth() {
  const [, setLocation] = useLocation();
  const { login, register } = useAuth();
  const { toast } = useToast();
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  
  // Form state
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [country, setCountry] = useState('India');
  const [allergies, setAllergies] = useState<string[]>([]);
  const [customAllergies, setCustomAllergies] = useState('');
  const [dislikedIngredients, setDislikedIngredients] = useState('');
  const [healthConditions, setHealthConditions] = useState('');
  const [cuisineType, setCuisineType] = useState('');
  const [dietaryPreferences, setDietaryPreferences] = useState('');
  const [productCategories, setProductCategories] = useState('');
  const [emergencyContact, setEmergencyContact] = useState('');

  const allergyOptions = [
    'Peanuts', 'Tree Nuts', 'Dairy', 'Gluten', 'Soy', 'Eggs', 'Fish', 'Shellfish', 'Sesame'
  ];

  const countryOptions = [
    'India', 'United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France', 'Other'
  ];

  const dietaryOptions = [
    'Vegetarian', 'Vegan', 'Keto', 'Paleo', 'Low-Carb', 'Gluten-Free', 'Halal', 'Kosher'
  ];

  const toggleAllergy = (allergy: string) => {
    setAllergies(prev => 
      prev.includes(allergy) 
        ? prev.filter(a => a !== allergy)
        : [...prev, allergy]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      let success = false;
      
      if (isLogin) {
        success = await login(email, password);
        if (success) {
          toast({
            title: "Welcome back!",
            description: "You have been logged in successfully.",
          });
          setLocation('/');
        } else {
          toast({
            title: "Login Failed",
            description: "Invalid email or password. Please try again.",
            variant: "destructive",
          });
        }
      } else {
        if (!firstName.trim() || !lastName.trim()) {
          toast({
            title: "Name Required",
            description: "Please enter your first and last name.",
            variant: "destructive",
          });
          return;
        }

        if (password !== confirmPassword) {
          toast({
            title: "Password Mismatch",
            description: "Passwords do not match.",
            variant: "destructive",
          });
          return;
        }

        // Combine predefined and custom allergies
        const allAllergies = [...allergies];
        if (customAllergies.trim()) {
          allAllergies.push(...customAllergies.split(',').map(item => item.trim()).filter(item => item));
        }
        
        success = await register({
          email,
          password,
          name: `${firstName.trim()} ${lastName.trim()}`,
          allergies: allAllergies,
          healthConditions: healthConditions.split(',').map(item => item.trim()).filter(item => item)
        });
        
        if (success) {
          toast({
            title: "Account Created!",
            description: "Welcome to FoodSense AI. Your health profile has been saved.",
          });
          setLocation('/');
        } else {
          toast({
            title: "Registration Failed",
            description: "Email may already exist. Please try a different email.",
            variant: "destructive",
          });
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary/10 to-secondary/10 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Back to Home */}
        <div className="mb-6">
          <Link href="/">
            <Button variant="ghost" className="flex items-center space-x-2">
              <ArrowLeft size={16} />
              <span>Back to Home</span>
            </Button>
          </Link>
        </div>

        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-gradient-to-r from-primary to-secondary rounded-2xl flex items-center justify-center mx-auto mb-4">
            <Leaf className="text-white" size={32} />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">FoodSense AI</h1>
          <p className="text-gray-600">Know What You Eat</p>
        </div>

        <Card className="shadow-xl">
          <CardHeader>
            <CardTitle className="text-center text-2xl">
              {isLogin ? 'Welcome Back!' : 'Create Your Allergy-Safe Account'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email */}
              <div className="space-y-2">
                <Label htmlFor="email" className="flex items-center space-x-2">
                  <Mail size={16} />
                  <span>Email Address</span>
                </Label>
                <Input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your email"
                  required
                />
              </div>

              {/* Password */}
              <div className="space-y-2">
                <Label htmlFor="password" className="flex items-center space-x-2">
                  <Lock size={16} />
                  <span>Password</span>
                </Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  required
                />
              </div>

              {/* Registration-only fields */}
              {!isLogin && (
                <>
                  {/* Confirm Password */}
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword" className="flex items-center space-x-2">
                      <Lock size={16} />
                      <span>Retype Password *</span>
                    </Label>
                    <Input
                      id="confirmPassword"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Confirm your password"
                      required
                    />
                  </div>

                  {/* Country */}
                  <div className="space-y-2">
                    <Label htmlFor="country">Country *</Label>
                    <Select value={country} onValueChange={setCountry}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select your country" />
                      </SelectTrigger>
                      <SelectContent>
                        {countryOptions.map((countryOption) => (
                          <SelectItem key={countryOption} value={countryOption}>
                            {countryOption}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* First Name */}
                  <div className="space-y-2">
                    <Label htmlFor="firstName" className="flex items-center space-x-2">
                      <User size={16} />
                      <span>First Name *</span>
                    </Label>
                    <Input
                      id="firstName"
                      type="text"
                      value={firstName}
                      onChange={(e) => setFirstName(e.target.value)}
                      placeholder="Enter your first name"
                      required
                    />
                  </div>

                  {/* Last Name */}
                  <div className="space-y-2">
                    <Label htmlFor="lastName">Last Name *</Label>
                    <Input
                      id="lastName"
                      type="text"
                      value={lastName}
                      onChange={(e) => setLastName(e.target.value)}
                      placeholder="Enter your last name"
                      required
                    />
                  </div>

                  <Separator className="my-6" />

                  {/* Customized Allergy Profile Setup */}
                  <div className="bg-gradient-to-r from-orange-50 to-red-50 p-6 rounded-xl border-l-4 border-orange-500">
                    <h4 className="font-bold text-gray-900 mb-4 flex items-center text-lg">
                      <AlertTriangle className="text-orange-500 mr-2" size={24} />
                      🍎 Customized Allergy Profile Setup
                    </h4>
                    
                    {/* Allergic Ingredients */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-3 block">
                        Allergic Ingredients (Required)
                      </Label>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
                        {allergyOptions.map((allergy) => (
                          <label 
                            key={allergy}
                            className="flex items-center space-x-2 p-3 border border-gray-200 rounded-lg hover:border-orange-400 transition-colors cursor-pointer bg-white"
                          >
                            <Checkbox 
                              checked={allergies.includes(allergy)}
                              onCheckedChange={() => toggleAllergy(allergy)}
                            />
                            <span className="text-sm font-medium">{allergy}</span>
                          </label>
                        ))}
                      </div>
                      <Textarea
                        placeholder="e.g., peanuts, soya, cashews"
                        value={customAllergies}
                        onChange={(e) => setCustomAllergies(e.target.value)}
                        className="mt-3"
                        rows={2}
                      />
                      <p className="text-xs text-gray-500 mt-1">Add other allergies not listed above, separated by commas</p>
                    </div>

                    {/* Disliked Ingredients */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Disliked Ingredients (Optional)
                      </Label>
                      <Textarea
                        placeholder="e.g., ghee, coriander"
                        value={dislikedIngredients}
                        onChange={(e) => setDislikedIngredients(e.target.value)}
                        rows={2}
                      />
                    </div>

                    {/* Health Conditions */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Health Conditions (Optional)
                      </Label>
                      <Textarea
                        placeholder="e.g., lactose intolerance, gluten sensitivity"
                        value={healthConditions}
                        onChange={(e) => setHealthConditions(e.target.value)}
                        rows={2}
                      />
                    </div>

                    {/* Preferred Cuisine Type */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Preferred Cuisine Type (Optional)
                      </Label>
                      <Input
                        placeholder="e.g., Indian, Vegan, Continental"
                        value={cuisineType}
                        onChange={(e) => setCuisineType(e.target.value)}
                      />
                    </div>

                    {/* Dietary Preferences */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Dietary Preferences *
                      </Label>
                      <Select value={dietaryPreferences} onValueChange={setDietaryPreferences}>
                        <SelectTrigger>
                          <SelectValue placeholder="--Select--" />
                        </SelectTrigger>
                        <SelectContent>
                          {dietaryOptions.map((option) => (
                            <SelectItem key={option} value={option}>
                              {option}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Preferred Product Categories */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Preferred Product Categories (Optional)
                      </Label>
                      <Input
                        placeholder="e.g., Snacks, Dairy, Beverages"
                        value={productCategories}
                        onChange={(e) => setProductCategories(e.target.value)}
                      />
                    </div>

                    {/* Emergency Contact */}
                    <div className="mb-6">
                      <Label className="text-sm font-semibold text-gray-700 mb-2 block">
                        Emergency Contact for Allergy Alerts (Optional)
                      </Label>
                      <Input
                        placeholder="Phone number or email"
                        value={emergencyContact}
                        onChange={(e) => setEmergencyContact(e.target.value)}
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Submit Button */}
              <Button 
                type="submit" 
                className={`w-full py-4 text-lg font-semibold ${
                  isLogin 
                    ? 'bg-primary text-white hover:bg-green-600' 
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
                disabled={isLoading}
              >
                {isLoading ? 'Please wait...' : (isLogin ? 'Login' : 'Create Account')}
              </Button>

              {/* Toggle Form */}
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => setIsLogin(!isLogin)}
                  className="text-primary hover:underline"
                >
                  {isLogin 
                    ? "Don't have an account? Create one" 
                    : "Already have an account? Login"
                  }
                </button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}