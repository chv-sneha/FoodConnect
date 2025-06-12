import { useState } from 'react';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { TopNavigation, BottomNavigation } from '@/components/navigation';
import { 
  Camera, 
  Shield, 
  Users, 
  Zap, 
  Languages, 
  Lightbulb,
  CheckCircle,
  ArrowRight
} from 'lucide-react';

export default function Home() {
  const [, setLocation] = useLocation();

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
                onClick={() => setLocation('/scan')}
              >
                <Camera className="mr-3" size={24} />
                Start Scanning
              </Button>
              <Button 
                variant="outline"
                size="lg"
                className="border-2 border-white text-white hover:bg-white hover:text-primary px-8 py-4 text-lg font-semibold transition-all duration-300"
              >
                Learn How It Works
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
              Why Choose FoodSense AI?
            </h2>
            <p className="text-xl text-gray-600">
              Advanced AI technology made simple for everyone
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow duration-300">
                <CardContent className="p-8 text-center">
                  <div className="w-16 h-16 bg-gradient-to-r from-primary to-secondary rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <feature.icon className="text-white" size={32} />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                  <p className="text-gray-600">{feature.description}</p>
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
              onClick={() => setLocation('/scan')}
            >
              Try Now <ArrowRight className="ml-2" />
            </Button>
            <Button 
              variant="outline"
              size="lg"
              className="border-2 border-white text-white hover:bg-white hover:text-primary px-8 py-4 text-lg font-semibold transition-all duration-300"
            >
              Download App
            </Button>
          </div>
        </div>
      </section>

      <BottomNavigation />
    </div>
  );
}
