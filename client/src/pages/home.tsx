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
  UserPlus
} from 'lucide-react';

export default function Home() {
  const [, setLocation] = useLocation();
  const { isAuthenticated } = useAuth();

  const features = [
    {
      icon: Camera,
      title: 'Instant Scanning',
      description: 'Just take a photo and get results in seconds using advanced OCR technology'
    },
    {
      icon: Shield,
      title: 'Traffic Light System',
      description: 'Easy-to-understand color coding: Green (Safe), Orange (Moderate), Red (Avoid)'
    },
    {
      icon: Users,
      title: 'Personal Alerts',
      description: 'Customized warnings based on your allergies, diabetes, and health conditions'
    },
    {
      icon: CheckCircle,
      title: 'FSSAI Verification',
      description: 'Automatically checks if products are legally approved and safe to consume'
    },
    {
      icon: Languages,
      title: 'Simple Language',
      description: 'Complex chemical names explained in plain language anyone can understand'
    },
    {
      icon: Lightbulb,
      title: 'Smart Suggestions',
      description: 'Get recommendations for healthier alternatives to products you scan'
    }
  ];

  const stats = [
    { value: '10M+', label: 'Products Analyzed' },
    { value: '500K+', label: 'Health Alerts Sent' },
    { value: '98%', label: 'Accuracy Rate' },
    { value: '1M+', label: 'Users Protected' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
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
                onClick={() => setLocation('/generic')}
              >
                <Search className="mr-3" size={24} />
                Generic Analysis
              </Button>
              <Button 
                variant="outline"
                size="lg"
                className="border-2 border-white text-white hover:bg-white hover:text-primary px-8 py-4 text-lg font-semibold transition-all duration-300"
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
            {/* Generic Analysis Card */}
            <Card className="hover:shadow-xl transition-all duration-300 border-2 hover:border-primary transform hover:scale-[1.02]">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute top-4 right-4 bg-gradient-to-r from-green-400 to-green-600 text-white px-3 py-1 rounded-full text-xs font-bold">
                  FREE
                </div>
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
                  onClick={() => setLocation('/generic')}
                >
                  Try Generic Analysis
                  <ArrowRight className="ml-2" size={20} />
                </Button>
              </CardContent>
            </Card>

            {/* Customized Analysis Card - Always Visible */}
            <Card className="hover:shadow-xl transition-all duration-300 border-2 hover:border-secondary transform hover:scale-[1.02]">
              <CardContent className="p-8 text-center relative overflow-hidden">
                {!isAuthenticated && (
                  <div className="absolute top-4 right-4 bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                    PREMIUM
                  </div>
                )}
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
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Why Choose FoodSense AI?</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow duration-300">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-gradient-to-r from-primary to-secondary rounded-xl flex items-center justify-center mx-auto mb-4">
                    <feature.icon className="text-white" size={24} />
                  </div>
                  <h4 className="text-lg font-bold text-gray-900 mb-3">{feature.title}</h4>
                  <p className="text-gray-600 text-sm">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-16 px-4 bg-gray-100">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-600">
              From scan to insights in just 3 simple steps
            </p>
          </div>

          <div className="space-y-12">
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="flex-shrink-0">
                <div className="w-20 h-20 bg-gradient-to-r from-primary to-secondary rounded-full flex items-center justify-center text-white text-2xl font-bold">
                  1
                </div>
              </div>
              <div className="flex-1">
                <img 
                  src="https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=800&h=600" 
                  alt="Person scanning food product with smartphone" 
                  className="rounded-2xl shadow-lg w-full h-64 object-cover mb-6" 
                />
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Scan or Upload</h3>
                <p className="text-lg text-gray-600">Take a photo of the back side of any packaged food product or upload an image from your gallery.</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row-reverse items-center gap-8">
              <div className="flex-shrink-0">
                <div className="w-20 h-20 bg-gradient-to-r from-secondary to-purple-500 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                  2
                </div>
              </div>
              <div className="flex-1">
                <img 
                  src="https://images.unsplash.com/photo-1677442136019-21780ecad995?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=800&h=600" 
                  alt="AI technology analyzing food data" 
                  className="rounded-2xl shadow-lg w-full h-64 object-cover mb-6" 
                />
                <h3 className="text-2xl font-bold text-gray-900 mb-4">AI Analysis</h3>
                <p className="text-lg text-gray-600">Our advanced AI extracts text, analyzes ingredients, checks for allergens, and verifies FSSAI compliance.</p>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="flex-shrink-0">
                <div className="w-20 h-20 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center text-white text-2xl font-bold">
                  3
                </div>
              </div>
              <div className="flex-1">
                <img 
                  src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=800&h=600" 
                  alt="Health dashboard with analysis results" 
                  className="rounded-2xl shadow-lg w-full h-64 object-cover mb-6" 
                />
                <h3 className="text-2xl font-bold text-gray-900 mb-4">Get Results</h3>
                <p className="text-lg text-gray-600">Receive instant health analysis with color-coded safety ratings and personalized alerts for your health conditions.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            {stats.map((stat, index) => (
              <div key={index} className="p-6">
                <div className="text-4xl md:text-5xl font-bold text-primary mb-2">{stat.value}</div>
                <p className="text-gray-600 text-lg">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="bg-gradient-to-r from-primary to-secondary text-white py-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Start Making Healthier Choices Today
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Join millions of users who trust FoodSense AI to keep them safe and informed
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              size="lg"
              className="bg-white text-primary hover:bg-gray-100 px-8 py-4 text-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300"
              onClick={() => setLocation('/generic')}
            >
              Try Generic Analysis <ArrowRight className="ml-2" />
            </Button>
            <Button 
              variant="outline"
              size="lg"
              className="border-2 border-white text-white hover:bg-white hover:text-primary px-8 py-4 text-lg font-semibold transition-all duration-300"
              onClick={() => isAuthenticated ? setLocation('/customized') : setLocation('/auth')}
            >
              Try Customized Analysis
            </Button>
          </div>
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}
