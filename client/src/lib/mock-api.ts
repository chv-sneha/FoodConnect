// Mock API for frontend-only deployment
export const mockAnalyzeGeneric = () => {
  return Promise.resolve({
    success: true,
    product_name: "Sample Food Product",
    final_safety_score: 72,
    ingredientAnalysis: [
      {
        ingredient: "Wheat Flour",
        safety_score: 8,
        risk_level: "Safe",
        reason: "Basic food ingredient, generally safe"
      },
      {
        ingredient: "Sugar",
        safety_score: 5,
        risk_level: "Medium",
        reason: "High sugar content, consume in moderation"
      }
    ],
    safety_analysis: {
      risk_summary: { High: 0, Medium: 1, Low: 0, Safe: 1 },
      ingredients: [
        { name: "Sugar", risk_level: "Medium", reason: "High sugar content" }
      ]
    },
    recommendations: {
      recommendations: ["Consume in moderation due to sugar content"]
    },
    timestamp: new Date().toISOString(),
    analysis_type: 'generic'
  });
};

export const mockAnalyzeCustomized = (userProfile: any) => {
  const detectedIngredients = ["Wheat Flour", "Sugar", "Palm Oil", "Salt", "Potato Starch"];
  const userAllergies = userProfile?.allergies || [];
  const userDisliked = userProfile?.dislikedIngredients || [];
  
  const personalizedWarnings: string[] = [];
  
  [...userAllergies, ...userDisliked].forEach((item: string) => {
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

  return Promise.resolve({
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
      recommendations: ["Check ingredients against your health profile"]
    },
    timestamp: new Date().toISOString(),
    analysis_type: 'customized'
  });
};