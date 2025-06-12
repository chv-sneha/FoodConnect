export interface AnalysisResult {
  id?: number;
  productName: string;
  safetyScore: 'safe' | 'moderate' | 'risky';
  analysis: {
    harmfulIngredients: HarmfulIngredient[];
    sugarLevel: 'low' | 'medium' | 'high';
    saltLevel: 'low' | 'medium' | 'high';
    additiveCount: number;
    preservativeCount: number;
  };
  ingredients: string[];
  fssaiVerified: boolean;
  fssaiNumber?: string;
  scannedAt?: Date;
}

export interface HarmfulIngredient {
  name: string;
  commonName?: string;
  description: string;
  concerns: string[];
}

export interface PersonalizedAlert {
  type: 'allergen' | 'health_condition';
  severity: 'low' | 'medium' | 'high';
  title: string;
  message: string;
}

export interface UserProfile {
  id: number;
  username: string;
  allergies: string[];
  healthConditions: string[];
}

export interface UploadResponse {
  success: boolean;
  productId?: number;
  error?: string;
}
