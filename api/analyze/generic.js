module.exports = (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const mockResult = {
    success: true,
    product_name: "Generic Food Product",
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
      },
      {
        ingredient: "Palm Oil",
        safety_score: 4,
        risk_level: "Medium",
        reason: "Processed oil, may contain trans fats"
      }
    ],
    safety_analysis: {
      risk_summary: { High: 0, Medium: 2, Low: 0, Safe: 1 },
      ingredients: [
        { name: "Sugar", risk_level: "Medium", reason: "High sugar content" },
        { name: "Palm Oil", risk_level: "Medium", reason: "Processed ingredient" }
      ]
    },
    recommendations: {
      recommendations: [
        "Consume in moderation due to sugar content",
        "Look for alternatives with natural sweeteners"
      ]
    },
    timestamp: new Date().toISOString(),
    analysis_type: 'generic'
  };

  res.json(mockResult);
}