export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const formData = req.body;
    const allergies = JSON.parse(formData.allergies || '[]');
    
    // Mock OCR text extraction
    const mockText = "wheat flour, sugar, milk powder, eggs, artificial flavoring";
    
    // Basic ingredient extraction
    const ingredients = mockText.split(',').map(i => i.trim());

    // Check for allergens
    const warnings = [];
    allergies.forEach(allergen => {
      if (mockText.toLowerCase().includes(allergen.toLowerCase())) {
        warnings.push(`Contains ${allergen} - may cause allergic reaction`);
      }
    });

    // Basic analysis
    const ingredientAnalysis = ingredients.map(ingredient => ({
      ingredient,
      toxicity_score: Math.random() * 100,
      health_impact: 'Moderate'
    }));

    return res.status(200).json({
      success: true,
      extracted_text: mockText,
      ingredientAnalysis,
      personalization: {
        warnings,
        recommendations: warnings.length === 0 ? ['Safe for consumption'] : ['Avoid this product']
      }
    });

  } catch (error) {
    return res.status(500).json({ 
      success: false, 
      error: 'Analysis failed' 
    });
  }
}