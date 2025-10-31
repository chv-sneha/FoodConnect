// Vercel serverless function for analysis
export default function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Mock analysis for now since Python won't work on Vercel
  const mockResult = {
    success: true,
    product_name: "Sample Food Product",
    final_safety_score: 75,
    ingredientAnalysis: [
      {
        ingredient: "Wheat Flour",
        safety_score: 8,
        risk_level: "Low",
        reason: "Basic food ingredient"
      },
      {
        ingredient: "Sugar",
        safety_score: 5,
        risk_level: "Medium", 
        reason: "High sugar content"
      }
    ],
    personalization: {
      warnings: [],
      recommendations: ["Consider healthier alternatives with less sugar"]
    },
    recommendations: {
      recommendations: ["Consume in moderation due to sugar content"]
    },
    timestamp: new Date().toISOString()
  };

  res.json(mockResult);
}