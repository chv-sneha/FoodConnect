import { HarmfulIngredient, AnalysisResult } from '@/types/analysis';

export function parseProductName(text: string): string {
  // Try to extract product name from the first line or packaging
  const lines = text.split('\n').filter(line => line.trim().length > 0);
  
  for (const line of lines.slice(0, 3)) {
    // Look for capitalized product names
    if (/^[A-Z][A-Za-z\s]{3,30}$/.test(line.trim())) {
      return line.trim();
    }
  }
  
  return 'Unknown Product';
}

export function extractNutritionalInfo(text: string): {
  sugar: string;
  salt: string;
  calories: string;
} {
  const nutritionalInfo = {
    sugar: 'Unknown',
    salt: 'Unknown',
    calories: 'Unknown'
  };

  // Extract sugar content
  const sugarMatch = text.match(/sugar[s]?\s*:?\s*([0-9.]+)\s*g/i);
  if (sugarMatch) {
    nutritionalInfo.sugar = `${sugarMatch[1]}g`;
  }

  // Extract salt content
  const saltMatch = text.match(/salt|sodium\s*:?\s*([0-9.]+)\s*g/i);
  if (saltMatch) {
    nutritionalInfo.salt = `${saltMatch[1]}g`;
  }

  // Extract calories
  const caloriesMatch = text.match(/calories?|energy\s*:?\s*([0-9.]+)/i);
  if (caloriesMatch) {
    nutritionalInfo.calories = `${caloriesMatch[1]} kcal`;
  }

  return nutritionalInfo;
}

export function validateFSSAINumber(fssaiNumber: string): boolean {
  // FSSAI numbers are 14 digits
  return /^[0-9]{14}$/.test(fssaiNumber);
}

export function getSafetyColor(score: 'safe' | 'moderate' | 'risky'): string {
  switch (score) {
    case 'safe':
      return 'text-green-600 bg-green-50 border-green-200';
    case 'moderate':
      return 'text-orange-600 bg-orange-50 border-orange-200';
    case 'risky':
      return 'text-red-600 bg-red-50 border-red-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
}

export function getSafetyIcon(score: 'safe' | 'moderate' | 'risky'): string {
  switch (score) {
    case 'safe':
      return 'fas fa-check-circle';
    case 'moderate':
      return 'fas fa-exclamation-triangle';
    case 'risky':
      return 'fas fa-exclamation-circle';
    default:
      return 'fas fa-question-circle';
  }
}

export function getLevelColor(level: 'low' | 'medium' | 'high'): string {
  switch (level) {
    case 'low':
      return 'text-green-600 bg-green-50';
    case 'medium':
      return 'text-orange-600 bg-orange-50';
    case 'high':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-gray-600 bg-gray-50';
  }
}
