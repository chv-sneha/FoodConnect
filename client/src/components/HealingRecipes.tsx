import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Heart, Search, Clock, Users } from "lucide-react";
import { BackButton } from "@/components/BackButton";

interface Recipe {
  name: string;
  condition: string;
  cookTime: string;
  servings: number;
  benefits: string[];
  ingredients: string[];
  calories: number;
  glycemicIndex?: number;
}

const healingRecipes: Recipe[] = [
  {
    name: "Fenugreek Dal",
    condition: "Diabetes",
    cookTime: "25 min",
    servings: 4,
    benefits: ["Controls blood sugar", "Rich in fiber", "Low GI"],
    ingredients: ["Toor dal", "Fenugreek seeds", "Turmeric", "Onions", "Tomatoes"],
    calories: 180,
    glycemicIndex: 35
  },
  {
    name: "Iron-Rich Spinach Curry",
    condition: "Anemia",
    cookTime: "20 min", 
    servings: 4,
    benefits: ["High iron content", "Vitamin C for absorption", "Folate rich"],
    ingredients: ["Fresh spinach", "Paneer", "Ginger", "Garlic", "Tomatoes"],
    calories: 220,
  },
  {
    name: "Coconut Laddu",
    condition: "Thyroid",
    cookTime: "15 min",
    servings: 6,
    benefits: ["Supports thyroid function", "Healthy fats", "Natural sweetness"],
    ingredients: ["Fresh coconut", "Jaggery", "Ghee", "Cardamom", "Almonds"],
    calories: 150,
  },
  {
    name: "Cinnamon Apple Oats",
    condition: "PCOD",
    cookTime: "12 min",
    servings: 2,
    benefits: ["Hormone balancing", "Anti-inflammatory", "Low insulin impact"],
    ingredients: ["Steel-cut oats", "Apple", "Cinnamon", "Walnuts", "Flaxseeds"],
    calories: 200,
    glycemicIndex: 42
  }
];

const conditions = ["All", "Diabetes", "PCOD", "Thyroid", "Anemia"];

const HealingRecipes = () => {
  const [selectedCondition, setSelectedCondition] = useState("All");
  const [searchTerm, setSearchTerm] = useState("");

  const filteredRecipes = healingRecipes.filter(recipe => {
    const matchesCondition = selectedCondition === "All" || recipe.condition === selectedCondition;
    const matchesSearch = recipe.name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCondition && matchesSearch;
  });

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        <BackButton />
        
        <div className="text-center mb-12 text-white">
          <Heart className="h-16 w-16 mx-auto mb-6" />
          <h1 className="text-5xl font-bold mb-4">Healing Recipes</h1>
          <p className="text-xl text-white/90 max-w-2xl mx-auto">
            Curated recipes for specific health conditions - tasty meals that heal
          </p>
        </div>

        <div className="max-w-6xl mx-auto space-y-8">
          <Card className="bg-white/95 backdrop-blur-sm">
            <CardContent className="p-6">
              <div className="flex flex-col md:flex-row gap-4 items-center">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search recipes..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
                <div className="flex flex-wrap gap-2">
                  {conditions.map((condition) => (
                    <Button
                      key={condition}
                      variant={selectedCondition === condition ? "default" : "outline"}
                      size="sm"
                      onClick={() => setSelectedCondition(condition)}
                      className={selectedCondition === condition ? "bg-gradient-card text-white" : ""}
                    >
                      {condition}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredRecipes.map((recipe, index) => (
              <Card key={index} className="bg-white/95 backdrop-blur-sm hover:shadow-medium transition-shadow">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-lg">{recipe.name}</CardTitle>
                    <Badge variant="secondary">{recipe.condition}</Badge>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {recipe.cookTime}
                    </div>
                    <div className="flex items-center gap-1">
                      <Users className="h-4 w-4" />
                      {recipe.servings} servings
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h4 className="font-semibold mb-2">Health Benefits</h4>
                    <div className="flex flex-wrap gap-1">
                      {recipe.benefits.map((benefit, i) => (
                        <Badge key={i} variant="outline" className="text-xs">
                          {benefit}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-2">Key Ingredients</h4>
                    <p className="text-sm text-muted-foreground">
                      {recipe.ingredients.join(", ")}
                    </p>
                  </div>

                  <div className="flex justify-between items-center pt-4 border-t">
                    <div className="text-sm">
                      <span className="font-semibold">{recipe.calories}</span> calories
                      {recipe.glycemicIndex && (
                        <span className="ml-2 text-muted-foreground">
                          GI: {recipe.glycemicIndex}
                        </span>
                      )}
                    </div>
                    <Button size="sm" className="bg-gradient-card text-white">
                      View Recipe
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {filteredRecipes.length === 0 && (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardContent className="text-center py-12">
                <p className="text-muted-foreground">
                  No recipes found for your search criteria.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default HealingRecipes;