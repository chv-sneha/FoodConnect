module.exports = (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Get user profile from request body
  const userAllergies = req.body.allergies ? JSON.parse(req.body.allergies) : [];
  const userDisliked = req.body.dislikedIngredients ? JSON.parse(req.body.dislikedIngredients) : [];
  const userConditions = req.body.healthConditions ? JSON.parse(req.body.healthConditions) : [];

  // Mock ingredients that might be detected
  const detectedIngredients = ["Wheat Flour", "Sugar", "Palm Oil", "Salt", "Potato Starch"];
  
  // Check for personalized warnings
  const personalizedWarnings = [];
  
  // Check allergies and disliked ingredients
  [...userAllergies, ...userDisliked].forEach(item => {
    const found = detectedIngredients.find(ing => 
      ing.toLowerCase().includes(item.toLowerCase())
    );
    if (found) {
      if (userAllergies.includes(item)) {
        personalizedWarnings.push(`⚠️ ALLERGEN ALERT: Contains ${item} - You're allergic!`);
      } else {
        personalizedWarnings.push(`⚠️ DISLIKED INGREDIENT: Contains ${item} - You prefer to avoid this`);
      }
    }
  });

  // Health condition warnings
  if (userConditions.includes('Diabetes') || userConditions.includes('diabetes')) {
    personalizedWarnings.push('🔴 High sugar content - May affect diabetes management');
  }

  const mockResult = {
    success: true,
    product_name: "Customized Food Product",
    final_safety_score: personalizedWarnings.length > 0 ? 45 : 75,
    ingredientAnalysis: detectedIngredients.map(ing => ({
      ingredient: ing,
      safety_score: ing === "Sugar" ? 4 : 7,
      risk_level: ing === "Sugar" ? "Medium" : "Safe",
      reason: ing === "Sugar" ? "High sugar content" : "Common food ingredient"
    })),
    personalization: {
      warnings: personalizedWarnings,
      recommendations: personalizedWarnings.length > 0 ? 
        ["Consider avoiding this product due to your health profile"] :
        ["This product appears safe for your profile"]
    },
    safety_analysis: {
      risk_summary: { High: personalizedWarnings.length, Medium: 1, Low: 0, Safe: 3 },
      ingredients: [
        { name: "Sugar", risk_level: "Medium", reason: "High sugar content" }
      ]
    },
    recommendations: {
      recommendations: [
        "Check ingredients against your health profile",
        "Consider healthier alternatives"
      ]
    },
    timestamp: new Date().toISOString(),
    analysis_type: 'customized'
  };

  res.json(mockResult);
}