import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Search, Clock, Users, ChefHat, Download, Share2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { BackButton } from "@/components/BackButton";

interface Recipe {
  name: string;
  ingredients: string[];
  cookTime: string;
  servings: number;
  category: string;
  difficulty: string;
  instructions?: string[];
  imageUrl?: string;
  nutrition?: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
}

const SmartRecipeList = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [filteredRecipes, setFilteredRecipes] = useState<Recipe[]>([]);
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [displayCount, setDisplayCount] = useState(20);
  const [sortBy, setSortBy] = useState('name');
  const [filterBy, setFilterBy] = useState('all');
  const { toast } = useToast();

  useEffect(() => {
    loadRecipes();
  }, []);

  useEffect(() => {
    let filtered = recipes;
    
    // Apply search filter
    if (searchTerm.trim()) {
      const searchLower = searchTerm.toLowerCase().trim();
      filtered = filtered.filter(recipe => {
        const nameMatch = recipe.name.toLowerCase().includes(searchLower);
        const ingredientMatch = recipe.ingredients.some(ingredient => 
          ingredient.toLowerCase().includes(searchLower)
        );
        const categoryMatch = recipe.category.toLowerCase().includes(searchLower);
        return nameMatch || ingredientMatch || categoryMatch;
      });
    }
    
    // Apply category filter
    if (filterBy !== 'all') {
      filtered = filtered.filter(recipe => 
        recipe.category.toLowerCase().includes(filterBy.toLowerCase())
      );
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'time':
          return parseInt(a.cookTime) - parseInt(b.cookTime);
        case 'ingredients':
          return a.ingredients.length - b.ingredients.length;
        case 'category':
          return a.category.localeCompare(b.category);
        default:
          return 0;
      }
    });
    
    setFilteredRecipes(filtered);
  }, [searchTerm, recipes, sortBy, filterBy]);

  const loadRecipes = async () => {
    try {
      setLoadingProgress(10);
      const response = await fetch('/recipes.csv');
      if (response.ok) {
        setLoadingProgress(30);
        const csvText = await response.text();
        setLoadingProgress(60);
        const parsedRecipes = parseCSV(csvText);
        setLoadingProgress(90);
        setRecipes(parsedRecipes);
        setLoadingProgress(100);
      } else {
        setRecipes(sampleRecipes);
      }
    } catch (error) {
      setRecipes(sampleRecipes);
    } finally {
      setLoading(false);
    }
  };

  const parseCSV = (csvText: string): Recipe[] => {
    const lines = csvText.split('\n');
    if (lines.length < 2) return [];
    
    const recipes: Recipe[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      
      try {
        const values = parseCSVLine(line);
        if (values.length < 6) continue;
        
        const name = values[0]?.replace(/^"|"$/g, '').trim();
        if (!name || name.length < 3) continue;
        
        const ingredients = values[1]?.replace(/^"|"$/g, '').trim();
        const time = values[2]?.trim() || '30';
        const cuisine = values[3]?.replace(/^"|"$/g, '').trim() || 'General';
        const instructions = values[4]?.replace(/^"|"$/g, '').trim();
        const imageUrl = values[5]?.replace(/^"|"$/g, '').trim();
        
        // Parse ingredients efficiently
        const cleanIngredients = ingredients ? 
          ingredients.split(',').map(i => i.trim()).filter(i => i.length > 2).slice(0, 15) : [];
        
        // Parse instructions efficiently
        const cleanInstructions = instructions ? 
          instructions.split(/\.|\n/).map(i => i.trim()).filter(i => i.length > 10).slice(0, 10) : [];
        
        recipes.push({
          name,
          ingredients: cleanIngredients,
          cookTime: `${time} min`,
          servings: 4,
          category: cuisine,
          difficulty: 'Medium',
          instructions: cleanInstructions,
          imageUrl: imageUrl || ''
        });
      } catch (error) {
        // Skip invalid lines silently for performance
        continue;
      }
    }
    
    return recipes;
  };

  const parseCSVLine = (line: string): string[] => {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current);
    return result;
  };

  const generateGroceryList = (recipe: Recipe) => {
    if (!recipe.ingredients || recipe.ingredients.length === 0) {
      toast({
        title: "No ingredients available",
        description: "This recipe doesn't have ingredient information",
        variant: "destructive"
      });
      return;
    }
    
    const groceryText = recipe.ingredients.map(ingredient => `• ${ingredient}`).join('\n');
    const content = `Grocery List for ${recipe.name}\n\nIngredients:\n${groceryText}\n\nCook Time: ${recipe.cookTime}\nServings: ${recipe.servings}\nCuisine: ${recipe.category}`;
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${recipe.name.replace(/\s+/g, '-')}-grocery-list.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Grocery list downloaded!",
      description: `Ingredients for ${recipe.name} saved as TXT file`
    });
  };

  const shareRecipe = async (recipe: Recipe) => {
    const ingredientsText = recipe.ingredients && recipe.ingredients.length > 0 
      ? recipe.ingredients.map(i => `• ${i}`).join('\n')
      : 'Ingredients not available';
    
    const text = `${recipe.name}\n\nIngredients:\n${ingredientsText}\n\nCook Time: ${recipe.cookTime}\nServings: ${recipe.servings}\nCuisine: ${recipe.category}`;
    
    if (navigator.share) {
      try {
        await navigator.share({ title: recipe.name, text });
      } catch {}
    } else {
      await navigator.clipboard.writeText(text);
      toast({ title: "Copied!", description: "Recipe copied to clipboard." });
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-hero flex items-center justify-center">
        <div className="text-white text-center">
          <ChefHat className="h-16 w-16 mx-auto mb-4 animate-pulse" />
          <p className="text-xl mb-4">Loading recipes...</p>
          <div className="w-64 bg-white/20 rounded-full h-2 mx-auto">
            <div 
              className="bg-white h-2 rounded-full transition-all duration-300" 
              style={{ width: `${loadingProgress}%` }}
            ></div>
          </div>
          <p className="text-sm mt-2 opacity-75">{loadingProgress}% complete</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-hero">
      <div className="container mx-auto px-6 py-20">
        <BackButton />
        
        <div className="text-center mb-12 text-white">
          <ChefHat className="h-16 w-16 mx-auto mb-6 animate-bounce" />
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-white to-orange-100 bg-clip-text text-transparent">Smart Recipe Collection</h1>
          <p className="text-xl text-white/90 max-w-2xl mx-auto leading-relaxed">
            Discover delicious recipes with ingredients and step-by-step cooking instructions from our collection of 50,000+ recipes
          </p>
        </div>

        <div className="max-w-7xl mx-auto space-y-8">
          <Card className="bg-white/95 backdrop-blur-sm shadow-lg border-0">
            <CardContent className="p-6 space-y-4">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-orange-500" />
                <Input
                  placeholder="Search recipes by name, ingredients, or cuisine... (e.g., 'chicken', 'biryani', 'South Indian')"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-12 py-3 text-lg border-2 border-orange-200 focus:border-orange-500 rounded-xl"
                />
              </div>
              
              {/* Filters and Sort */}
              <div className="flex flex-wrap gap-4 items-center">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">Sort by:</label>
                  <select 
                    value={sortBy} 
                    onChange={(e) => setSortBy(e.target.value)}
                    className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="name">Name</option>
                    <option value="time">Cook Time</option>
                    <option value="ingredients">Ingredients Count</option>
                    <option value="category">Category</option>
                  </select>
                </div>
                
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">Filter:</label>
                  <select 
                    value={filterBy} 
                    onChange={(e) => setFilterBy(e.target.value)}
                    className="px-3 py-1 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="all">All Cuisines</option>
                    <option value="indian">Indian</option>
                    <option value="south indian">South Indian</option>
                    <option value="north indian">North Indian</option>
                    <option value="chinese">Chinese</option>
                    <option value="continental">Continental</option>
                    <option value="mexican">Mexican</option>
                    <option value="thai">Thai</option>
                  </select>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-600 flex items-center gap-2">
                  <ChefHat className="h-4 w-4 text-orange-500" />
                  Found <span className="font-semibold text-orange-600">{filteredRecipes.length}</span> delicious recipes
                </p>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <span>Showing {Math.min(displayCount, filteredRecipes.length)} of {filteredRecipes.length}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredRecipes.slice(0, displayCount).map((recipe, index) => (
              <Card key={index} className="bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300 border-0 shadow-md hover:scale-105 cursor-pointer group">
                <CardContent className="p-0">
                  {/* Recipe Image */}
                  <div className="relative mb-4 rounded-t-lg overflow-hidden h-48 bg-gradient-to-br from-orange-100 to-red-100">
                    {recipe.imageUrl ? (
                      <img 
                        src={recipe.imageUrl} 
                        alt={recipe.name}
                        className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none';
                        }}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-orange-100 to-red-100">
                        <ChefHat className="h-16 w-16 text-orange-300" />
                      </div>
                    )}
                    
                    {/* Cuisine Badge - Top Right */}
                    <div className="absolute top-3 right-3">
                      <div className="bg-gradient-to-r from-orange-500 to-red-600 text-white font-medium px-3 py-1 text-xs rounded-full shadow-lg">
                        {recipe.category}
                      </div>
                    </div>
                    
                    {/* Cooking Time - Bottom Left */}
                    <div className="absolute bottom-3 left-3">
                      <div className="bg-black/70 backdrop-blur-sm rounded-full px-3 py-1 flex items-center gap-1">
                        <Clock className="h-3 w-3 text-white" />
                        <span className="text-white text-xs font-medium">{recipe.cookTime}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="px-4 pb-4">
                    {/* Recipe Name */}
                    <h3 className="text-lg font-bold text-gray-800 leading-tight mb-2 min-h-[3.5rem] overflow-hidden" style={{display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
                      {recipe.name}
                    </h3>
                    
                    {/* Ingredients Count */}
                    <div className="flex items-center gap-1 mb-3 text-orange-600">
                      <ChefHat className="h-4 w-4" />
                      <span className="text-sm font-medium">{recipe.ingredients.length} ingredients</span>
                    </div>

                    {/* Recipe Stats */}
                    <div className="flex items-center justify-between mb-4 text-gray-600">
                      <div className="flex items-center gap-1">
                        <Clock className="h-4 w-4 text-orange-500" />
                        <span className="text-sm">{recipe.cookTime}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Users className="h-4 w-4 text-orange-500" />
                        <span className="text-sm">{recipe.servings} servings</span>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2">
                      <Button 
                        className="flex-1 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-semibold py-2 text-sm rounded-lg transition-all duration-300 shadow-md hover:shadow-lg"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRecipe(recipe);
                        }}
                      >
                        View Recipe
                      </Button>
                      <Button 
                        variant="outline"
                        size="sm"
                        className="px-3 border-2 border-orange-200 hover:border-orange-400 hover:bg-orange-50 transition-all duration-300 rounded-lg"
                        onClick={(e) => {
                          e.stopPropagation();
                          generateGroceryList(recipe);
                        }}
                      >
                        <Download className="h-4 w-4 text-orange-600" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {filteredRecipes.length === 0 && searchTerm && (
            <Card className="bg-white/95 backdrop-blur-sm">
              <CardContent className="text-center py-12">
                <ChefHat className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-muted-foreground">
                  No recipes found for "{searchTerm}". Try searching for something else.
                </p>
              </CardContent>
            </Card>
          )}
          
          {filteredRecipes.length > displayCount && (
            <div className="text-center">
              <Button 
                onClick={() => setDisplayCount(prev => prev + 20)}
                className="bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-semibold px-8 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
              >
                Load More Recipes
                <span className="ml-2 bg-white/20 px-2 py-1 rounded-full text-sm">
                  +{Math.min(20, filteredRecipes.length - displayCount)}
                </span>
              </Button>
              <p className="text-sm text-gray-500 mt-2">
                {filteredRecipes.length - displayCount} more recipes available
              </p>
            </div>
          )}
        </div>

        {selectedRecipe && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <Card className="max-w-4xl w-full max-h-[90vh] overflow-y-auto bg-white shadow-2xl">
              <CardHeader className="bg-gradient-to-r from-orange-500 to-red-600 text-white">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <CardTitle className="text-3xl font-bold mb-2">{selectedRecipe.name}</CardTitle>
                    <div className="flex items-center gap-4 text-orange-100">
                      <Badge className="bg-white/20 text-white border-white/30">
                        {selectedRecipe.category}
                      </Badge>
                      <div className="flex items-center gap-1">
                        <Clock className="h-4 w-4" />
                        <span>{selectedRecipe.cookTime}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Users className="h-4 w-4" />
                        <span>{selectedRecipe.servings} servings</span>
                      </div>
                    </div>
                  </div>
                  <Button 
                    variant="ghost" 
                    onClick={() => setSelectedRecipe(null)}
                    className="text-white hover:bg-white/20 text-2xl font-bold w-10 h-10 p-0"
                  >
                    ×
                  </Button>
                </div>
              </CardHeader>
              
              <CardContent className="p-8 space-y-8">
                {/* Recipe Image */}
                {selectedRecipe.imageUrl && (
                  <div className="rounded-xl overflow-hidden shadow-lg">
                    <img 
                      src={selectedRecipe.imageUrl} 
                      alt={selectedRecipe.name}
                      className="w-full h-64 object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  </div>
                )}

                {/* Ingredients Section */}
                {selectedRecipe.ingredients && selectedRecipe.ingredients.length > 0 && (
                  <div className="bg-orange-50 rounded-xl p-6">
                    <h4 className="text-2xl font-bold mb-6 text-gray-800 flex items-center gap-2">
                      <ChefHat className="h-6 w-6 text-orange-600" />
                      Ingredients ({selectedRecipe.ingredients.length})
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedRecipe.ingredients.map((ingredient, i) => (
                        <div key={i} className="flex items-start p-4 bg-white rounded-lg shadow-sm border border-orange-100 hover:shadow-md transition-shadow">
                          <span className="w-3 h-3 bg-orange-500 rounded-full mr-3 flex-shrink-0 mt-1"></span>
                          <span className="text-gray-700 leading-relaxed">{ingredient}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Instructions Section */}
                {selectedRecipe.instructions && selectedRecipe.instructions.length > 0 && (
                  <div className="bg-blue-50 rounded-xl p-6">
                    <h4 className="text-2xl font-bold mb-6 text-gray-800 flex items-center gap-2">
                      <ChefHat className="h-6 w-6 text-blue-600" />
                      Cooking Instructions
                    </h4>
                    <ol className="space-y-4">
                      {selectedRecipe.instructions.map((step, i) => (
                        <li key={i} className="flex gap-4 p-4 bg-white rounded-lg shadow-sm border border-blue-100 hover:shadow-md transition-shadow">
                          <span className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-full w-10 h-10 flex items-center justify-center text-sm font-bold flex-shrink-0">
                            {i + 1}
                          </span>
                          <div className="flex-1">
                            <p className="text-gray-700 leading-relaxed">{step}</p>
                          </div>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Show message if no ingredients or instructions */}
                {(!selectedRecipe.ingredients || selectedRecipe.ingredients.length === 0) && 
                 (!selectedRecipe.instructions || selectedRecipe.instructions.length === 0) && (
                  <div className="text-center py-12 text-gray-500 bg-gray-50 rounded-xl">
                    <ChefHat className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                    <p className="text-lg font-medium mb-2">Recipe Details Loading...</p>
                    <p className="text-sm">Please wait while we fetch the complete recipe information.</p>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-4 pt-4">
                  <Button 
                    className="flex-1 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white font-semibold py-3 text-lg rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
                    onClick={() => generateGroceryList(selectedRecipe)}
                  >
                    <Download className="h-5 w-5 mr-2" />
                    Download Grocery List
                  </Button>
                  <Button 
                    variant="outline"
                    className="flex-1 border-2 border-orange-300 hover:border-orange-500 hover:bg-orange-50 text-orange-700 font-semibold py-3 text-lg rounded-xl transition-all duration-300"
                    onClick={() => shareRecipe(selectedRecipe)}
                  >
                    <Share2 className="h-5 w-5 mr-2" />
                    Share Recipe
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

// Sample recipes as fallback
const sampleRecipes: Recipe[] = [
  {
    name: "Butter Chicken",
    ingredients: ["Chicken breast", "Butter", "Tomato puree", "Heavy cream", "Garam masala", "Ginger-garlic paste"],
    cookTime: "45 min",
    servings: 4,
    category: "Indian",
    difficulty: "Medium",
    instructions: ["Marinate chicken with spices", "Cook chicken until tender", "Prepare creamy tomato sauce", "Combine chicken with sauce", "Simmer and serve hot"]
  },
  {
    name: "Chicken Curry",
    ingredients: ["Chicken", "Onions", "Tomatoes", "Coconut milk", "Curry powder", "Garlic"],
    cookTime: "40 min",
    servings: 4,
    category: "Indian",
    difficulty: "Medium",
    instructions: ["Sauté onions and garlic", "Add chicken and brown", "Add tomatoes and spices", "Pour coconut milk and simmer", "Cook until chicken is tender"]
  },
  {
    name: "Chicken Biryani",
    ingredients: ["Chicken", "Basmati rice", "Yogurt", "Biryani masala", "Saffron", "Mint leaves"],
    cookTime: "90 min",
    servings: 6,
    category: "Indian",
    difficulty: "Hard",
    instructions: ["Marinate chicken with yogurt and spices", "Cook rice with whole spices", "Layer rice and chicken", "Add saffron and mint", "Cook on dum for 45 minutes"]
  },
  {
    name: "Vegetable Biryani",
    ingredients: ["Basmati rice", "Mixed vegetables", "Yogurt", "Biryani masala", "Saffron", "Mint leaves"],
    cookTime: "60 min",
    servings: 6,
    category: "Indian",
    difficulty: "Hard",
    instructions: ["Prepare vegetables with spices", "Cook rice with aromatics", "Layer rice and vegetables", "Add saffron and herbs", "Cook on dum for 30 minutes"]
  }
];

export default SmartRecipeList;