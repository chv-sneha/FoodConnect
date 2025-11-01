export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Mock generic analysis
    const mockIngredients = [
      'wheat flour', 'sugar', 'vegetable oil', 'salt', 'baking powder'
    ];

    const ingredientAnalysis = mockIngredients.map(ingredient => ({
      ingredient,
      toxicity_score: Math.floor(Math.random() * 100),
      health_impact: Math.random() > 0.5 ? 'Low' : 'Moderate',
      category: 'processed'
    }));

    return res.status(200).json({
      success: true,
      extracted_text: mockIngredients.join(', '),
      ingredientAnalysis,
      overall_score: Math.floor(Math.random() * 100),
      recommendations: [
        'Consider products with fewer processed ingredients',
        'Look for whole grain alternatives'
      ]
    });

  } catch (error) {
    return res.status(500).json({ 
      success: false, 
      error: 'Analysis failed' 
    });
  }
}