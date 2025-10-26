import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Heart, Activity } from "lucide-react";
import { BackButton } from "@/components/BackButton";

interface HealthMetric {
  name: string;
  current: number;
  trend: "up" | "down" | "stable";
  risk: "low" | "medium" | "high";
  prediction: string;
  recommendation: string;
}

const healthData: HealthMetric[] = [
  {
    name: "Blood Sugar Risk",
    current: 65,
    trend: "up",
    risk: "medium",
    prediction: "15% increase in diabetes risk if current pattern continues",
    recommendation: "Reduce sugar intake by 40%, add more fiber-rich foods"
  },
  {
    name: "Heart Health Score",
    current: 78,
    trend: "stable",
    risk: "low",
    prediction: "Good cardiovascular health maintenance",
    recommendation: "Continue current healthy eating pattern"
  },
  {
    name: "Nutrient Balance", 
    current: 85,
    trend: "up",
    risk: "low",
    prediction: "Excellent nutritional profile trending positively",
    recommendation: "Maintain diverse food choices"
  },
  {
    name: "Inflammation Markers",
    current: 45,
    trend: "down",
    risk: "medium", 
    prediction: "Reducing inflammation levels - good progress",
    recommendation: "Add more anti-inflammatory foods like turmeric, ginger"
  }
];

const AiHealthForecast = () => {
  const [selectedMetric, setSelectedMetric] = useState<HealthMetric | null>(null);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low": return "text-green-600";
      case "medium": return "text-yellow-600"; 
      case "high": return "text-red-600";
      default: return "text-gray-600";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "up": return <TrendingUp className="h-4 w-4 text-green-600" />;
      case "down": return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <Activity className="h-4 w-4 text-blue-600" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        <BackButton />
        
        <div className="text-center mb-12 text-white">
          <TrendingUp className="h-16 w-16 mx-auto mb-6" />
          <h1 className="text-5xl font-bold mb-4">AI Health Forecast</h1>
          <p className="text-xl text-white/90 max-w-2xl mx-auto">
            Predict future health risks from your eating patterns and prevent them early
          </p>
        </div>

        <div className="max-w-6xl mx-auto space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {healthData.map((metric, index) => (
              <Card 
                key={index} 
                className="bg-white/95 backdrop-blur-sm cursor-pointer hover:shadow-medium transition-all"
                onClick={() => setSelectedMetric(metric)}
              >
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-sm">{metric.name}</h3>
                    {getTrendIcon(metric.trend)}
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold">{metric.current}%</span>
                      <Badge 
                        variant="outline" 
                        className={getRiskColor(metric.risk)}
                      >
                        {metric.risk}
                      </Badge>
                    </div>
                    <Progress value={metric.current} className="h-2" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {selectedMetric && (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Heart className="h-5 w-5 text-red-500" />
                  {selectedMetric.name} - Detailed Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                      AI Prediction
                    </h4>
                    <p className="text-muted-foreground bg-yellow-50 p-4 rounded-lg">
                      {selectedMetric.prediction}
                    </p>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-600" />
                      Recommended Actions
                    </h4>
                    <p className="text-muted-foreground bg-green-50 p-4 rounded-lg">
                      {selectedMetric.recommendation}
                    </p>
                  </div>
                </div>

                <div className="border-t pt-6">
                  <h4 className="font-semibold mb-4">Weekly Progress Tracking</h4>
                  <div className="grid grid-cols-7 gap-2">
                    {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((day, i) => (
                      <div key={day} className="text-center">
                        <div className="text-xs text-muted-foreground mb-1">{day}</div>
                        <div className={`h-8 rounded ${i < 5 ? 'bg-green-200' : 'bg-gray-200'}`}></div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card className="bg-white/95 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Health Score Gamification</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 border rounded-lg">
                  <div className="text-3xl font-bold text-green-600 mb-2">7</div>
                  <div className="text-sm text-muted-foreground">Day Streak</div>
                  <Badge className="mt-2 bg-green-100 text-green-800">Healthy Choices</Badge>
                </div>
                <div className="text-center p-6 border rounded-lg">
                  <div className="text-3xl font-bold text-blue-600 mb-2">750</div>
                  <div className="text-sm text-muted-foreground">Health Points</div>
                  <Badge className="mt-2 bg-blue-100 text-blue-800">Wellness Champion</Badge>
                </div>
                <div className="text-center p-6 border rounded-lg">
                  <div className="text-3xl font-bold text-purple-600 mb-2">12</div>
                  <div className="text-sm text-muted-foreground">Badges Earned</div>
                  <Badge className="mt-2 bg-purple-100 text-purple-800">Nutrition Expert</Badge>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="text-center">
            <Button size="lg" className="bg-gradient-card text-white">
              Generate Full Health Report
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AiHealthForecast;