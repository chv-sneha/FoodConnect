import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertUserSchema, insertScannedProductSchema, loginSchema } from "@shared/schema";
import multer from "multer";
import path from "path";

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|webp/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  
  // User registration
  app.post("/api/auth/register", async (req, res) => {
    try {
      const userData = insertUserSchema.parse(req.body);
      const existingUser = await storage.getUserByEmail(userData.email);
      
      if (existingUser) {
        return res.status(400).json({ message: "Email already exists" });
      }
      
      const user = await storage.createUser(userData);
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // User login
  app.post("/api/auth/login", async (req, res) => {
    try {
      const loginData = loginSchema.parse(req.body);
      const user = await storage.getUserByEmail(loginData.email);
      
      if (!user || user.password !== loginData.password) {
        return res.status(401).json({ message: "Invalid email or password" });
      }
      
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Get user profile
  app.get("/api/users/:id", async (req, res) => {
    try {
      const userId = parseInt(req.params.id);
      const user = await storage.getUser(userId);
      
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Update user profile (allergies and health conditions)
  app.put("/api/users/:id/profile", async (req, res) => {
    try {
      const userId = parseInt(req.params.id);
      const { allergies, healthConditions } = req.body;
      
      const user = await storage.updateUserProfile(userId, allergies || [], healthConditions || []);
      
      if (!user) {
        return res.status(404).json({ message: "User not found" });
      }
      
      const { password, ...userWithoutPassword } = user;
      res.json(userWithoutPassword);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Upload and analyze product image
  app.post("/api/products/analyze", upload.single('image'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ message: "No image file provided" });
      }

      const { userId, extractedText, productName } = req.body;
      
      if (!extractedText || !productName) {
        return res.status(400).json({ message: "Extracted text and product name are required" });
      }

      // Parse ingredients from extracted text
      const ingredients = parseIngredients(extractedText);
      
      // Analyze ingredients
      const analysis = await analyzeIngredients(ingredients);
      
      // Determine safety score
      const safetyScore = determineSafetyScore(analysis);
      
      // Check FSSAI verification (mock implementation)
      const { fssaiVerified, fssaiNumber } = checkFSSAIVerification(extractedText);

      const productData = {
        userId: userId ? parseInt(userId) : null,
        productName,
        imageUrl: `/uploads/${req.file.filename}`,
        extractedText,
        ingredients,
        analysis,
        safetyScore,
        fssaiVerified,
        fssaiNumber
      };

      const scannedProduct = await storage.createScannedProduct(productData);
      res.json(scannedProduct);
      
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  // Get user's scanned products history
  app.get("/api/users/:id/products", async (req, res) => {
    try {
      const userId = parseInt(req.params.id);
      const products = await storage.getScannedProductsByUser(userId);
      res.json(products);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Get specific product analysis
  app.get("/api/products/:id", async (req, res) => {
    try {
      const productId = parseInt(req.params.id);
      const product = await storage.getScannedProduct(productId);
      
      if (!product) {
        return res.status(404).json({ message: "Product not found" });
      }
      
      res.json(product);
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Get personalized analysis for a product based on user allergies
  app.post("/api/products/:id/personalized", async (req, res) => {
    try {
      const productId = parseInt(req.params.id);
      const { userId } = req.body;
      
      const product = await storage.getScannedProduct(productId);
      const user = await storage.getUser(userId);
      
      if (!product || !user) {
        return res.status(404).json({ message: "Product or user not found" });
      }

      const personalizedAlerts = generatePersonalizedAlerts(
        product.ingredients || [],
        user.allergies || [],
        user.healthConditions || []
      );

      res.json({ personalizedAlerts });
    } catch (error: any) {
      res.status(400).json({ message: error.message });
    }
  });

  // Get all ingredients database
  app.get("/api/ingredients", async (req, res) => {
    try {
      const ingredients = await storage.getAllIngredients();
      res.json(ingredients);
    } catch (error: any) {
      res.status(500).json({ message: error.message });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

// Helper functions
function parseIngredients(text: string): string[] {
  // Extract ingredients section from the text
  const ingredientsMatch = text.match(/ingredients?[:\s]*(.*?)(?=\n|nutritional|nutrition|contains|allergen|$)/i);
  
  if (!ingredientsMatch) return [];
  
  const ingredientsText = ingredientsMatch[1];
  
  // Split by common separators and clean up
  return ingredientsText
    .split(/[,;]/)
    .map(ingredient => ingredient.trim())
    .filter(ingredient => ingredient.length > 0)
    .map(ingredient => ingredient.replace(/^\W+|\W+$/g, ''));
}

async function analyzeIngredients(ingredients: string[]) {
  const analysis = {
    harmfulIngredients: [],
    sugarLevel: 'low',
    saltLevel: 'low',
    additiveCount: 0,
    preservativeCount: 0
  };

  let sugarContent = 0;
  let saltContent = 0;

  for (const ingredient of ingredients) {
    const knownIngredient = await storage.getIngredientByName(ingredient);
    
    if (knownIngredient && knownIngredient.riskLevel === 'high') {
      analysis.harmfulIngredients.push({
        name: knownIngredient.name,
        commonName: knownIngredient.commonName,
        description: knownIngredient.description,
        concerns: knownIngredient.concerns
      });
    }

    // Check for additives (E-numbers)
    if (/E\d{3,4}|INS\s?\d{3,4}/i.test(ingredient)) {
      analysis.additiveCount++;
    }

    // Check for preservatives
    if (/preservative|sodium benzoate|potassium sorbate|citric acid/i.test(ingredient)) {
      analysis.preservativeCount++;
    }

    // Estimate sugar content
    if (/sugar|fructose|glucose|sucrose|corn syrup|honey|molasses/i.test(ingredient)) {
      sugarContent += 10; // Rough estimation
    }

    // Estimate salt content
    if (/sodium|salt|sea salt/i.test(ingredient)) {
      saltContent += 5; // Rough estimation
    }
  }

  // Determine levels
  analysis.sugarLevel = sugarContent > 15 ? 'high' : sugarContent > 8 ? 'medium' : 'low';
  analysis.saltLevel = saltContent > 10 ? 'high' : saltContent > 5 ? 'medium' : 'low';

  return analysis;
}

function determineSafetyScore(analysis: any): string {
  const { harmfulIngredients, sugarLevel, saltLevel, additiveCount } = analysis;
  
  let riskScore = 0;
  
  if (harmfulIngredients.length > 0) riskScore += 3;
  if (sugarLevel === 'high') riskScore += 2;
  if (saltLevel === 'high') riskScore += 1;
  if (additiveCount > 3) riskScore += 2;
  
  if (riskScore >= 4) return 'risky';
  if (riskScore >= 2) return 'moderate';
  return 'safe';
}

function checkFSSAIVerification(text: string): { fssaiVerified: boolean; fssaiNumber: string | null } {
  const fssaiMatch = text.match(/fssai[:\s]*([0-9]{14})/i);
  
  if (fssaiMatch) {
    return {
      fssaiVerified: true,
      fssaiNumber: fssaiMatch[1]
    };
  }
  
  return {
    fssaiVerified: false,
    fssaiNumber: null
  };
}

function generatePersonalizedAlerts(
  ingredients: string[], 
  allergies: string[], 
  healthConditions: string[]
): any[] {
  const alerts = [];
  
  // Check for allergens
  for (const allergy of allergies) {
    for (const ingredient of ingredients) {
      if (ingredient.toLowerCase().includes(allergy.toLowerCase())) {
        alerts.push({
          type: 'allergen',
          severity: 'high',
          title: 'Allergen Warning',
          message: `Contains ${allergy} which you're allergic to. May cause allergic reactions.`
        });
      }
    }
  }
  
  // Check for health condition conflicts
  for (const condition of healthConditions) {
    if (condition.toLowerCase() === 'diabetes') {
      const sugarIngredients = ingredients.filter(ing => 
        /sugar|fructose|glucose|corn syrup/i.test(ing)
      );
      if (sugarIngredients.length > 0) {
        alerts.push({
          type: 'health_condition',
          severity: 'medium',
          title: 'Diabetic Alert',
          message: 'This product contains high sugar content. Not recommended for diabetics.'
        });
      }
    }
    
    if (condition.toLowerCase() === 'hypertension') {
      const saltIngredients = ingredients.filter(ing => 
        /sodium|salt/i.test(ing)
      );
      if (saltIngredients.length > 0) {
        alerts.push({
          type: 'health_condition',
          severity: 'medium',
          title: 'Hypertension Alert',
          message: 'This product contains high sodium content. May affect blood pressure.'
        });
      }
    }
  }
  
  return alerts;
}
