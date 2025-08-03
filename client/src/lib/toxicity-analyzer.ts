// Toxicity Scoring Engine
export interface ToxicityAnalysis {
  overallScore: number; // 0-100 (0 = safest, 100 = most toxic)
  chemicalLoad: number;
  sugarContent: number;
  saltContent: number;
  additiveCount: number;
  preservativeCount: number;
  harmfulIngredients: string[];
  riskFactors: string[];
  recommendations: string[];
}

interface IngredientToxicity {
  name: string;
  toxicityScore: number;
  category: 'chemical' | 'sugar' | 'salt' | 'additive' | 'preservative' | 'natural';
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  concerns: string[];
}

// Comprehensive toxicity database
const TOXICITY_DATABASE: Record<string, IngredientToxicity> = {
  // High-risk chemicals and additives
  'msg': { name: 'MSG', toxicityScore: 85, category: 'additive', riskLevel: 'high', concerns: ['Headaches', 'Nausea', 'Neural effects'] },
  'monosodium glutamate': { name: 'MSG', toxicityScore: 85, category: 'additive', riskLevel: 'high', concerns: ['Headaches', 'Nausea', 'Neural effects'] },
  'aspartame': { name: 'Aspartame', toxicityScore: 90, category: 'additive', riskLevel: 'extreme', concerns: ['Cancer risk', 'Neural toxicity', 'Headaches'] },
  'sucralose': { name: 'Sucralose', toxicityScore: 75, category: 'additive', riskLevel: 'high', concerns: ['Gut microbiome disruption', 'Glucose metabolism issues'] },
  'acesulfame potassium': { name: 'Acesulfame K', toxicityScore: 80, category: 'additive', riskLevel: 'high', concerns: ['Cancer concerns', 'Insulin response'] },
  'sodium benzoate': { name: 'Sodium Benzoate', toxicityScore: 70, category: 'preservative', riskLevel: 'medium', concerns: ['DNA damage when combined with vitamin C', 'Hyperactivity'] },
  'potassium sorbate': { name: 'Potassium Sorbate', toxicityScore: 60, category: 'preservative', riskLevel: 'medium', concerns: ['Allergic reactions', 'DNA mutations'] },
  'bht': { name: 'BHT', toxicityScore: 95, category: 'preservative', riskLevel: 'extreme', concerns: ['Cancer risk', 'Hormone disruption', 'Liver damage'] },
  'bha': { name: 'BHA', toxicityScore: 95, category: 'preservative', riskLevel: 'extreme', concerns: ['Cancer risk', 'Endocrine disruption'] },
  'tbhq': { name: 'TBHQ', toxicityScore: 85, category: 'preservative', riskLevel: 'high', concerns: ['Nausea', 'Vomiting', 'Ringing in ears'] },
  'red 40': { name: 'Red 40', toxicityScore: 75, category: 'additive', riskLevel: 'high', concerns: ['Hyperactivity in children', 'Allergic reactions'] },
  'yellow 6': { name: 'Yellow 6', toxicityScore: 75, category: 'additive', riskLevel: 'high', concerns: ['Hyperactivity', 'Cancer concerns'] },
  'blue 1': { name: 'Blue 1', toxicityScore: 70, category: 'additive', riskLevel: 'medium', concerns: ['Allergic reactions', 'Chromosome damage'] },
  'caramel color': { name: 'Caramel Color', toxicityScore: 65, category: 'additive', riskLevel: 'medium', concerns: ['Cancer risk from 4-MEI compound'] },
  
  // Trans fats and harmful oils
  'trans fat': { name: 'Trans Fat', toxicityScore: 100, category: 'chemical', riskLevel: 'extreme', concerns: ['Heart disease', 'Stroke', 'Diabetes'] },
  'partially hydrogenated': { name: 'Partially Hydrogenated Oil', toxicityScore: 95, category: 'chemical', riskLevel: 'extreme', concerns: ['Trans fats', 'Heart disease'] },
  'hydrogenated oil': { name: 'Hydrogenated Oil', toxicityScore: 90, category: 'chemical', riskLevel: 'extreme', concerns: ['Trans fats', 'Cardiovascular disease'] },
  'palm oil': { name: 'Palm Oil', toxicityScore: 40, category: 'natural', riskLevel: 'low', concerns: ['High saturated fat', 'Environmental impact'] },
  
  // Sugars and sweeteners
  'high fructose corn syrup': { name: 'High Fructose Corn Syrup', toxicityScore: 80, category: 'sugar', riskLevel: 'high', concerns: ['Obesity', 'Diabetes', 'Liver damage'] },
  'corn syrup': { name: 'Corn Syrup', toxicityScore: 70, category: 'sugar', riskLevel: 'medium', concerns: ['Blood sugar spikes', 'Weight gain'] },
  'sugar': { name: 'Sugar', toxicityScore: 50, category: 'sugar', riskLevel: 'medium', concerns: ['Diabetes', 'Obesity', 'Tooth decay'] },
  'glucose': { name: 'Glucose', toxicityScore: 45, category: 'sugar', riskLevel: 'medium', concerns: ['Blood sugar spikes'] },
  'fructose': { name: 'Fructose', toxicityScore: 55, category: 'sugar', riskLevel: 'medium', concerns: ['Liver fat accumulation', 'Insulin resistance'] },
  'sucrose': { name: 'Sucrose', toxicityScore: 50, category: 'sugar', riskLevel: 'medium', concerns: ['Diabetes risk', 'Tooth decay'] },
  
  // Sodium and salts
  'sodium': { name: 'Sodium', toxicityScore: 60, category: 'salt', riskLevel: 'medium', concerns: ['High blood pressure', 'Heart disease'] },
  'salt': { name: 'Salt', toxicityScore: 55, category: 'salt', riskLevel: 'medium', concerns: ['Hypertension', 'Kidney strain'] },
  'sodium chloride': { name: 'Sodium Chloride', toxicityScore: 55, category: 'salt', riskLevel: 'medium', concerns: ['Blood pressure', 'Water retention'] },
  
  // Natural ingredients (low toxicity)
  'water': { name: 'Water', toxicityScore: 0, category: 'natural', riskLevel: 'low', concerns: [] },
  'wheat flour': { name: 'Wheat Flour', toxicityScore: 20, category: 'natural', riskLevel: 'low', concerns: ['Gluten intolerance'] },
  'milk': { name: 'Milk', toxicityScore: 15, category: 'natural', riskLevel: 'low', concerns: ['Lactose intolerance'] },
  'eggs': { name: 'Eggs', toxicityScore: 10, category: 'natural', riskLevel: 'low', concerns: ['Allergen'] },
  'cocoa': { name: 'Cocoa', toxicityScore: 10, category: 'natural', riskLevel: 'low', concerns: ['Caffeine content'] },
  'vanilla': { name: 'Vanilla', toxicityScore: 5, category: 'natural', riskLevel: 'low', concerns: [] },
};

// Healthy ingredient substitutes
const INGREDIENT_SUBSTITUTES: Record<string, string[]> = {
  'sugar': ['Jaggery', 'Honey', 'Dates', 'Stevia', 'Monk fruit'],
  'high fructose corn syrup': ['Pure maple syrup', 'Coconut sugar', 'Date syrup'],
  'palm oil': ['Coconut oil', 'Olive oil', 'Avocado oil', 'Sunflower oil'],
  'msg': ['Natural herbs', 'Garlic powder', 'Onion powder', 'Nutritional yeast'],
  'sodium benzoate': ['Vitamin E', 'Rosemary extract', 'Citric acid'],
  'artificial colors': ['Turmeric (yellow)', 'Beetroot (red)', 'Spirulina (blue)'],
  'trans fat': ['Coconut oil', 'Ghee', 'Olive oil'],
  'aspartame': ['Stevia', 'Monk fruit', 'Erythritol'],
};

export function analyzeToxicity(ingredients: string[]): ToxicityAnalysis {
  let totalToxicity = 0;
  let chemicalLoad = 0;
  let sugarContent = 0;
  let saltContent = 0;
  let additiveCount = 0;
  let preservativeCount = 0;
  const harmfulIngredients: string[] = [];
  const riskFactors: string[] = [];
  const recommendations: string[] = [];

  // Analyze each ingredient
  ingredients.forEach(ingredient => {
    const normalizedIngredient = ingredient.toLowerCase().trim();
    
    // Check exact matches first
    let toxicityData = TOXICITY_DATABASE[normalizedIngredient];
    
    // Check partial matches for compound ingredients
    if (!toxicityData) {
      for (const [key, data] of Object.entries(TOXICITY_DATABASE)) {
        if (normalizedIngredient.includes(key) || key.includes(normalizedIngredient)) {
          toxicityData = data;
          break;
        }
      }
    }

    if (toxicityData) {
      totalToxicity += toxicityData.toxicityScore;
      
      // Categorize by type
      switch (toxicityData.category) {
        case 'chemical':
          chemicalLoad += toxicityData.toxicityScore;
          break;
        case 'sugar':
          sugarContent += toxicityData.toxicityScore;
          break;
        case 'salt':
          saltContent += toxicityData.toxicityScore;
          break;
        case 'additive':
          additiveCount++;
          break;
        case 'preservative':
          preservativeCount++;
          break;
      }

      // Add to harmful ingredients if high risk
      if (toxicityData.riskLevel === 'high' || toxicityData.riskLevel === 'extreme') {
        harmfulIngredients.push(toxicityData.name);
        riskFactors.push(...toxicityData.concerns);
        
        // Add substitution recommendations
        const substitutes = INGREDIENT_SUBSTITUTES[normalizedIngredient];
        if (substitutes) {
          recommendations.push(`Replace ${toxicityData.name} with: ${substitutes.join(', ')}`);
        }
      }
    }
  });

  // Calculate overall score (0-100)
  const overallScore = Math.min(100, Math.round(totalToxicity / Math.max(1, ingredients.length)));

  // Add general recommendations based on scores
  if (sugarContent > 150) {
    recommendations.push('Reduce sugar intake - consider natural sweeteners like jaggery or dates');
  }
  if (saltContent > 100) {
    recommendations.push('High sodium content - limit portion size and drink extra water');
  }
  if (additiveCount > 3) {
    recommendations.push('Too many additives - choose products with simpler ingredient lists');
  }
  if (chemicalLoad > 200) {
    recommendations.push('High chemical load - opt for organic or naturally preserved alternatives');
  }

  return {
    overallScore,
    chemicalLoad: Math.min(100, Math.round(chemicalLoad / 3)),
    sugarContent: Math.min(100, Math.round(sugarContent / 2)),
    saltContent: Math.min(100, Math.round(saltContent / 2)),
    additiveCount,
    preservativeCount,
    harmfulIngredients: [...new Set(harmfulIngredients)], // Remove duplicates
    riskFactors: [...new Set(riskFactors)], // Remove duplicates
    recommendations: [...new Set(recommendations)] // Remove duplicates
  };
}

export function getToxicityScoreColor(score: number): string {
  if (score <= 30) return 'text-green-600';
  if (score <= 60) return 'text-yellow-600';
  if (score <= 80) return 'text-orange-600';
  return 'text-red-600';
}

export function getToxicityScoreLabel(score: number): string {
  if (score <= 30) return 'Low Toxicity';
  if (score <= 60) return 'Moderate Toxicity';
  if (score <= 80) return 'High Toxicity';
  return 'Very High Toxicity';
}