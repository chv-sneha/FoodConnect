import { useState } from 'react';
import { useLocation } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { useAuth } from '@/context/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { 
  User, 
  Mail, 
  Lock, 
  AlertTriangle, 
  Heart, 
  Leaf,
  ArrowLeft
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
  const [name, setName] = useState('');
  const [allergies, setAllergies] = useState<string[]>([]);
  const [healthConditions, setHealthConditions] = useState<string[]>([]);

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
        if (!name.trim()) {
          toast({
            title: "Name Required",
            description: "Please enter your name.",
            variant: "destructive",
          });
          return;
        }
        
        success = await register({
          email,
          password,
          name: name.trim(),
          allergies,
          healthConditions
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
            <CardTitle className="text-center">
              {isLogin ? 'Login to Your Account' : 'Create Your Account'}
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
                  {/* Name */}
                  <div className="space-y-2">
                    <Label htmlFor="name" className="flex items-center space-x-2">
                      <User size={16} />
                      <span>Full Name</span>
                    </Label>
                    <Input
                      id="name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Enter your full name"
                      required
                    />
                  </div>

                  <Separator />

                  {/* Health Profile Section */}
                  <div>
                    <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                      <AlertTriangle className="text-orange-500 mr-2" size={20} />
                      Your Allergies & Intolerances
                    </h4>
                    <div className="grid grid-cols-2 gap-3 mb-6">
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

                    <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                      <Heart className="text-red-500 mr-2" size={20} />
                      Health Conditions
                    </h4>
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
                </>
              )}

              {/* Submit Button */}
              <Button 
                type="submit" 
                className="w-full bg-primary text-white hover:bg-green-600 py-4 text-lg font-semibold"
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