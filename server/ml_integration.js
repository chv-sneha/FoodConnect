const { spawn } = require('child_process');
const path = require('path');

class MLIntegration {
  constructor() {
    this.modelPath = path.join(__dirname, '../ml_models/models/production_model.pkl');
    this.pythonScript = path.join(__dirname, '../ml_models/predict.py');
  }

  async analyzeIngredients(ingredients) {
    return new Promise((resolve, reject) => {
      // Create Python script to run ML prediction
      const pythonProcess = spawn('python', [this.pythonScript, JSON.stringify(ingredients)]);
      
      let result = '';
      let error = '';

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const predictions = JSON.parse(result);
            resolve(predictions);
          } catch (e) {
            reject(new Error('Failed to parse ML predictions'));
          }
        } else {
          reject(new Error(`ML prediction failed: ${error}`));
        }
      });
    });
  }

  // Fallback ingredient analysis using our database
  analyzeIngredientsBasic(ingredients) {
    const ingredientDatabase = {
      'sugar': { toxicity: 65, category: 'sweetener', risk: 'medium' },
      'salt': { toxicity: 60, category: 'preservative', risk: 'medium' },
      'trans fat': { toxicity: 95, category: 'fat', risk: 'high' },
      'msg': { toxicity: 85, category: 'flavor_enhancer', risk: 'high' },
      'aspartame': { toxicity: 80, category: 'sweetener', risk: 'high' },
      'sodium benzoate': { toxicity: 75, category: 'preservative', risk: 'high' },
      'high fructose corn syrup': { toxicity: 90, category: 'sweetener', risk: 'high' },
      'artificial colors': { toxicity: 75, category: 'additive', risk: 'high' },
      'vitamin c': { toxicity: 5, category: 'vitamin', risk: 'safe' },
      'calcium': { toxicity: 0, category: 'mineral', risk: 'safe' },
      'protein': { toxicity: 0, category: 'macronutrient', risk: 'safe' },
      'fiber': { toxicity: 0, category: 'fiber', risk: 'safe' }
    };

    const results = ingredients.map(ingredient => {
      const cleanIngredient = ingredient.toLowerCase().trim();
      const match = Object.keys(ingredientDatabase).find(key => 
        cleanIngredient.includes(key) || key.includes(cleanIngredient)
      );

      if (match) {
        const data = ingredientDatabase[match];
        return {
          ingredient: ingredient,
          toxicity_score: data.toxicity,
          category: data.category,
          risk_level: data.risk,
          is_toxic: data.toxicity > 50,
          confidence: 0.85
        };
      }

      // Default for unknown ingredients
      return {
        ingredient: ingredient,
        toxicity_score: 30,
        category: 'unknown',
        risk_level: 'low',
        is_toxic: false,
        confidence: 0.3
      };
    });

    return results;
  }

  calculateOverallScore(ingredientAnalysis) {
    if (!ingredientAnalysis || ingredientAnalysis.length === 0) {
      return { score: 50, grade: 'C', safety: 'moderate' };
    }

    const totalToxicity = ingredientAnalysis.reduce((sum, item) => sum + item.toxicity_score, 0);
    const avgToxicity = totalToxicity / ingredientAnalysis.length;
    
    // Convert toxicity to health score (inverse)
    const healthScore = Math.max(0, 100 - avgToxicity);
    
    // Calculate grade
    let grade = 'C';
    if (healthScore >= 80) grade = 'A';
    else if (healthScore >= 65) grade = 'B';
    else if (healthScore >= 50) grade = 'C';
    else if (healthScore >= 35) grade = 'D';
    else grade = 'E';

    // Safety level
    let safety = 'moderate';
    if (avgToxicity < 30) safety = 'safe';
    else if (avgToxicity < 50) safety = 'moderate';
    else if (avgToxicity < 70) safety = 'risky';
    else safety = 'dangerous';

    return {
      score: Math.round(healthScore),
      grade: grade,
      safety: safety,
      avgToxicity: Math.round(avgToxicity),
      totalIngredients: ingredientAnalysis.length,
      toxicIngredients: ingredientAnalysis.filter(i => i.is_toxic).length
    };
  }

  generateRecommendations(ingredientAnalysis, userProfile = null) {
    const recommendations = [];
    const toxicIngredients = ingredientAnalysis.filter(i => i.is_toxic);

    if (toxicIngredients.length > 0) {
      recommendations.push({
        type: 'warning',
        message: `Found ${toxicIngredients.length} concerning ingredients`,
        ingredients: toxicIngredients.map(i => i.ingredient)
      });
    }

    // Suggest alternatives for high-risk ingredients
    const alternatives = {
      'sugar': ['stevia', 'honey', 'maple syrup'],
      'salt': ['herbs', 'spices', 'lemon juice'],
      'trans fat': ['olive oil', 'coconut oil'],
      'msg': ['natural herbs', 'garlic powder'],
      'artificial colors': ['natural coloring', 'turmeric']
    };

    toxicIngredients.forEach(item => {
      const ingredient = item.ingredient.toLowerCase();
      const alt = Object.keys(alternatives).find(key => ingredient.includes(key));
      if (alt) {
        recommendations.push({
          type: 'substitute',
          message: `Consider replacing ${item.ingredient} with ${alternatives[alt].join(', ')}`,
          original: item.ingredient,
          alternatives: alternatives[alt]
        });
      }
    });

    // Personalized recommendations based on user profile
    if (userProfile) {
      if (userProfile.allergies) {
        const allergenMatches = ingredientAnalysis.filter(item => 
          userProfile.allergies.some(allergy => 
            item.ingredient.toLowerCase().includes(allergy.toLowerCase())
          )
        );
        
        if (allergenMatches.length > 0) {
          recommendations.push({
            type: 'allergen_alert',
            message: 'ALLERGEN ALERT: This product contains ingredients you\'re allergic to',
            allergens: allergenMatches.map(i => i.ingredient)
          });
        }
      }

      if (userProfile.healthConditions) {
        if (userProfile.healthConditions.includes('diabetes')) {
          const sugarIngredients = ingredientAnalysis.filter(item => 
            ['sugar', 'glucose', 'fructose', 'corn syrup'].some(sugar => 
              item.ingredient.toLowerCase().includes(sugar)
            )
          );
          if (sugarIngredients.length > 0) {
            recommendations.push({
              type: 'health_warning',
              message: 'Diabetes Alert: High sugar content detected',
              concern: 'diabetes',
              ingredients: sugarIngredients.map(i => i.ingredient)
            });
          }
        }

        if (userProfile.healthConditions.includes('hypertension')) {
          const saltIngredients = ingredientAnalysis.filter(item => 
            ['salt', 'sodium', 'monosodium'].some(salt => 
              item.ingredient.toLowerCase().includes(salt)
            )
          );
          if (saltIngredients.length > 0) {
            recommendations.push({
              type: 'health_warning',
              message: 'Hypertension Alert: High sodium content detected',
              concern: 'hypertension',
              ingredients: saltIngredients.map(i => i.ingredient)
            });
          }
        }
      }
    }

    return recommendations;
  }
}

module.exports = MLIntegration;