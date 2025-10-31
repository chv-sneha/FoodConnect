import { useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { useAuth } from '@/context/AuthContext';
import { 
  Camera, 
  Shield, 
  Users, 
  Zap, 
  Languages, 
  Lightbulb,
  CheckCircle,
  ArrowRight,
  Search,
  UserPlus,
  ShoppingCart,
  DollarSign,
  ChefHat,
  Heart,
  Brain
} from 'lucide-react';

export default function Home() {
  const [, setLocation] = useLocation();
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: ChefHat,
      title: 'Smart Recipe List',
      description: 'Search thousands of recipes and generate instant grocery lists for your cooking needs.'
    },
    {
      icon: DollarSign,
      title: 'AI Meal Budget Planner',
      description: 'Eat healthy without breaking the bank – smart meal plans within your budget.'
    },
    {
      icon: ChefHat,
      title: 'Meal Prep & Planner',
      description: 'Weekly meal prep schedules with portion guidance, so you never miss a healthy meal.'
    },
    {
      icon: Heart,
      title: 'Healing Recipes',
      description: 'Curated recipes for PCOD, diabetes, thyroid, anemia & more – tasty meals that heal.'
    },
    {
      icon: Brain,
      title: 'AI Health Forecast',
      description: 'Predict future health risks from your eating patterns – and prevent them early.'
    },
    {
      icon: Shield,
      title: 'Food Safety & Consumer Rights',
      description: 'Stay protected with alerts on food fraud, adulteration, legal approvals & your consumer rights.'
    }
  ];



  return (
    <div className="min-h-screen bg-gray-50 relative overflow-hidden">
      {/* Animated background div */}
      <div
        className="slide-in-top-normal"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          zIndex: 0,
          background: 'linear-gradient(120deg, #e0e7ff 0%, #f0fdfa 100%)',
          opacity: 0.5,
        }}
      />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <TopNavigation />
        {/* Hero Section */}
        <section className="bg-gradient-to-r from-primary to-secondary text-white py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <div className="animate-in fade-in duration-700">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight">
              Know What You're<br />
              <span className="text-yellow-300">Really Eating</span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100 leading-relaxed">
              Upload any food product image and get instant safety analysis, allergen warnings, and ingredient breakdown in simple language.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button 
                size="lg"
                className="bg-white text-primary hover:bg-gray-100 px-8 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
                onClick={() => isAuthenticated ? setLocation('/generic') : setLocation('/auth')}
              >
                <Search className="mr-3" size={24} />
                Generic Analysis
              </Button>
              <Button 
                size="lg"
                className="bg-white text-primary hover:bg-gray-100 px-8 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
                onClick={() => isAuthenticated ? setLocation('/customized') : setLocation('/auth')}
              >
                <UserPlus className="mr-3" size={24} />
                Customized Analysis
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Choose Your Analysis Type
            </h2>
            <p className="text-xl text-gray-600">
              Two powerful ways to understand your food products
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 mb-12 md:mb-16">
            {/* Generic Analysis Card */}
            <Card className="hover:shadow-xl transition-all duration-300 border-2 hover:border-primary transform hover:scale-[1.02]">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="w-20 h-20 bg-gradient-to-r from-primary to-green-400 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <Search className="text-white" size={40} />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Generic Analysis</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  Perfect for everyone! Upload any food product image and get instant ingredient breakdown, toxicity analysis, sugar/salt levels, and FSSAI verification. No registration required.
                </p>
                
                {/* Feature Highlights */}
                <div className="mb-6 space-y-2 text-left bg-green-50 p-4 rounded-lg">
                  <div className="flex items-center text-sm text-green-800">
                    <CheckCircle size={16} className="mr-2 text-green-600" />
                    <span>Ingredient breakdown</span>
                  </div>
                  <div className="flex items-center text-sm text-green-800">
                    <CheckCircle size={16} className="mr-2 text-green-600" />
                    <span>Safety scoring system</span>
                  </div>
                  <div className="flex items-center text-sm text-green-800">
                    <CheckCircle size={16} className="mr-2 text-green-600" />
                    <span>FSSAI verification</span>
                  </div>
                </div>

                <Button 
                  className="w-full bg-primary text-white hover:bg-green-600 py-3 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
                  onClick={() => isAuthenticated ? setLocation('/generic') : setLocation('/auth')}
                >
                  {isAuthenticated ? 'Try Generic Analysis' : 'Login & Try Generic Analysis'}
                  <ArrowRight className="ml-2" size={20} />
                </Button>
              </CardContent>
            </Card>

            {/* Customized Analysis Card - Always Visible */}
            <Card className="hover:shadow-xl transition-all duration-300 border-2 hover:border-secondary transform hover:scale-[1.02]">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="w-20 h-20 bg-gradient-to-r from-secondary to-blue-400 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <UserPlus className="text-white" size={40} />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Customized Analysis</h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  Personalized for you! Set your allergies, health conditions, and dietary preferences. Get instant alerts when products contain ingredients that could harm you.
                </p>
                
                {/* Feature Highlights */}
                <div className="mb-6 space-y-2 text-left bg-blue-50 p-4 rounded-lg">
                  <div className="flex items-center text-sm text-blue-800">
                    <CheckCircle size={16} className="mr-2 text-blue-600" />
                    <span>Personalized allergen warnings</span>
                  </div>
                  <div className="flex items-center text-sm text-blue-800">
                    <CheckCircle size={16} className="mr-2 text-blue-600" />
                    <span>Health condition alerts</span>
                  </div>
                  <div className="flex items-center text-sm text-blue-800">
                    <CheckCircle size={16} className="mr-2 text-blue-600" />
                    <span>Custom dietary tracking</span>
                  </div>
                </div>

                <Button 
                  className="w-full bg-secondary text-white hover:bg-blue-600 py-3 text-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
                  onClick={() => isAuthenticated ? setLocation('/customized') : setLocation('/auth')}
                >
                  {isAuthenticated ? 'Start Personalized Scan' : 'Create Account & Start'}
                  <ArrowRight className="ml-2" size={20} />
                </Button>
              </CardContent>
            </Card>
          </div>

          <div className="text-center mb-12">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Why Choose Food Connect?</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const getFeatureRoute = (title: string) => {
                switch (title) {
                  case 'Smart Recipe List': return '/smart-recipe-list';
                  case 'AI Meal Budget Planner': return '/ai-meal-budget-planner';
                  case 'Meal Prep & Planner': return '/meal-prep-planner';
                  case 'Healing Recipes': return '/healing-recipes';
                  case 'AI Health Forecast': return '/ai-health-forecast';
                  case 'Food Safety & Consumer Rights': return '/consumer-rights';
                  default: return '/';
                }
              };
              
              return (
                <Card 
                  key={index} 
                  className="hover:shadow-lg transition-all duration-300 cursor-pointer transform hover:scale-[1.02]"
                  onClick={() => setLocation(getFeatureRoute(feature.title))}
                >
                  <CardContent className="p-6 text-center">
                    <div className="w-12 h-12 bg-gradient-to-r from-primary to-secondary rounded-xl flex items-center justify-center mx-auto mb-4">
                      <feature.icon className="text-white" size={24} />
                    </div>
                    <h4 className="text-lg font-bold text-gray-900 mb-3">{feature.title}</h4>
                    <p className="text-gray-600 text-sm">{feature.description}</p>
                    <div className="mt-4">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="hover:bg-primary hover:text-white transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          setLocation(getFeatureRoute(feature.title));
                        }}
                      >
                        Try Now <ArrowRight className="ml-1" size={14} />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>



      {/* Call to Action Section */}
      <section className="bg-gradient-to-r from-primary to-secondary text-white py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
          </div>
          <div className="pt-8">
            <p className="text-white mb-2">
              © 2025 Food Connect. All rights reserved.
            </p>
            <p className="text-sm text-gray-200">
              Empowering healthier food choices through AI technology.
            </p>
          </div>
        </div>
      </section>

        <BottomNavigation />
      </div>
    </div>
  );
}
