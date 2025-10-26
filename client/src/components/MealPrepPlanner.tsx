import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Calendar, Clock, Users, Bell } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { BackButton } from "@/components/BackButton";

interface MealSlot {
  time: string;
  meal: string;
  portions: string;
  prepTime: string;
}

const weeklySchedule = {
  Monday: [
    { time: "8:00 AM", meal: "Overnight Oats", portions: "4 servings", prepTime: "5 min prep" },
    { time: "1:00 PM", meal: "Dal Tadka + Rice", portions: "4 servings", prepTime: "25 min" },
    { time: "8:00 PM", meal: "Roti + Mixed Veg Curry", portions: "4 servings", prepTime: "30 min" }
  ],
  Tuesday: [
    { time: "8:00 AM", meal: "Poha with Vegetables", portions: "4 servings", prepTime: "15 min" },
    { time: "1:00 PM", meal: "Rajma + Rice", portions: "4 servings", prepTime: "35 min" },
    { time: "8:00 PM", meal: "Chapati + Palak Paneer", portions: "4 servings", prepTime: "25 min" }
  ],
  Wednesday: [
    { time: "8:00 AM", meal: "Upma with Nuts", portions: "4 servings", prepTime: "12 min" },
    { time: "1:00 PM", meal: "Chole + Roti", portions: "4 servings", prepTime: "30 min" },
    { time: "8:00 PM", meal: "Rice + Sambar", portions: "4 servings", prepTime: "28 min" }
  ]
};

const MealPrepPlanner = () => {
  const [selectedDay, setSelectedDay] = useState("Monday");
  const [portionSize, setPortionSize] = useState(4);
  const { toast } = useToast();

  const scheduleReminder = (meal: string, time: string) => {
    toast({
      title: "Reminder Set!",
      description: `You'll be reminded to prepare ${meal} at ${time}`
    });
  };

  const generateGroceryList = () => {
    toast({
      title: "Grocery list generated!",
      description: "Your weekly shopping list is ready based on meal plan"
    });
  };

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        <BackButton />
        
        <div className="text-center mb-12 text-white">
          <Calendar className="h-16 w-16 mx-auto mb-6" />
          <h1 className="text-5xl font-bold mb-4">Meal Prep & Planner</h1>
          <p className="text-xl text-white/90 max-w-2xl mx-auto">
            Weekly meal scheduling with smart reminders and portion guidance
          </p>
        </div>

        <div className="max-w-6xl mx-auto space-y-8">
          <Card className="bg-white/95 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Weekly Meal Calendar</span>
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  <Badge variant="secondary">{portionSize} people</Badge>
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 mb-6">
                {Object.keys(weeklySchedule).map((day) => (
                  <Button
                    key={day}
                    variant={selectedDay === day ? "default" : "outline"}
                    onClick={() => setSelectedDay(day)}
                    className={selectedDay === day ? "bg-gradient-card text-white" : ""}
                  >
                    {day}
                  </Button>
                ))}
              </div>

              <div className="space-y-4">
                {weeklySchedule[selectedDay as keyof typeof weeklySchedule].map((slot, index) => (
                  <div key={index} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <Clock className="h-4 w-4 text-muted-foreground" />
                          <span className="font-medium">{slot.time}</span>
                          <Badge variant="outline">{slot.prepTime}</Badge>
                        </div>
                        <h4 className="font-semibold text-lg">{slot.meal}</h4>
                        <p className="text-muted-foreground">{slot.portions}</p>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => scheduleReminder(slot.meal, slot.time)}
                        className="flex items-center gap-2"
                      >
                        <Bell className="h-4 w-4" />
                        Set Reminder
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Prep Tips for {selectedDay}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                    <span>Soak lentils overnight for faster cooking</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                    <span>Prep vegetables in the morning to save time</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 bg-primary rounded-full mt-2"></div>
                    <span>Use pressure cooker for dal to reduce cook time</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/95 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button 
                  onClick={generateGroceryList}
                  className="w-full bg-gradient-card text-white"
                >
                  Generate Weekly Grocery List
                </Button>
                <Button variant="outline" className="w-full">
                  Export Meal Schedule
                </Button>
                <Button variant="outline" className="w-full">
                  Share with Family
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MealPrepPlanner;