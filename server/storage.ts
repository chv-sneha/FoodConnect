import { 
  users, 
  scannedProducts, 
  ingredients,
  type User, 
  type InsertUser,
  type ScannedProduct,
  type InsertScannedProduct,
  type Ingredient,
  type InsertIngredient
} from "@shared/schema";

export interface IStorage {
  // User operations
  getUser(id: number): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  updateUserProfile(id: number, allergies: string[], healthConditions: string[]): Promise<User | undefined>;
  
  // Product scanning operations
  createScannedProduct(product: InsertScannedProduct): Promise<ScannedProduct>;
  getScannedProductsByUser(userId: number): Promise<ScannedProduct[]>;
  getScannedProduct(id: number): Promise<ScannedProduct | undefined>;
  
  // Ingredient operations
  getIngredientByName(name: string): Promise<Ingredient | undefined>;
  getAllIngredients(): Promise<Ingredient[]>;
  createIngredient(ingredient: InsertIngredient): Promise<Ingredient>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private scannedProducts: Map<number, ScannedProduct>;
  private ingredients: Map<number, Ingredient>;
  private currentUserId: number;
  private currentProductId: number;
  private currentIngredientId: number;

  constructor() {
    this.users = new Map();
    this.scannedProducts = new Map();
    this.ingredients = new Map();
    this.currentUserId = 1;
    this.currentProductId = 1;
    this.currentIngredientId = 1;
    
    // Initialize with common harmful ingredients
    this.initializeIngredients();
  }

  private async initializeIngredients() {
    const commonIngredients = [
      {
        name: "Sodium Benzoate",
        commonName: "E211",
        description: "Artificial preservative that may cause allergic reactions",
        riskLevel: "high" as const,
        allergenType: "preservative",
        concerns: ["allergic reactions", "hyperactivity"]
      },
      {
        name: "High Fructose Corn Syrup",
        commonName: "HFCS",
        description: "Linked to obesity and metabolic disorders",
        riskLevel: "high" as const,
        allergenType: "sugar",
        concerns: ["obesity", "diabetes", "metabolic disorders"]
      },
      {
        name: "Monosodium Glutamate",
        commonName: "MSG",
        description: "Flavor enhancer that may cause headaches",
        riskLevel: "medium" as const,
        allergenType: "flavor enhancer",
        concerns: ["headaches", "nausea"]
      },
      {
        name: "Aspartame",
        commonName: "E951",
        description: "Artificial sweetener with potential health risks",
        riskLevel: "medium" as const,
        allergenType: "artificial sweetener",
        concerns: ["headaches", "mood changes"]
      },
      {
        name: "Trans Fat",
        commonName: "Partially Hydrogenated Oil",
        description: "Unhealthy fat linked to heart disease",
        riskLevel: "high" as const,
        allergenType: "fat",
        concerns: ["heart disease", "cholesterol"]
      }
    ];

    for (const ingredient of commonIngredients) {
      await this.createIngredient(ingredient);
    }
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.email === email,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { 
      ...insertUser, 
      id,
      allergies: insertUser.allergies || [],
      healthConditions: insertUser.healthConditions || [],
      createdAt: new Date()
    };
    this.users.set(id, user);
    return user;
  }

  async updateUserProfile(id: number, allergies: string[], healthConditions: string[]): Promise<User | undefined> {
    const user = this.users.get(id);
    if (!user) return undefined;
    
    const updatedUser = { ...user, allergies, healthConditions };
    this.users.set(id, updatedUser);
    return updatedUser;
  }

  async createScannedProduct(product: InsertScannedProduct): Promise<ScannedProduct> {
    const id = this.currentProductId++;
    const scannedProduct: ScannedProduct = {
      ...product,
      id,
      ingredients: product.ingredients || [],
      scannedAt: new Date()
    };
    this.scannedProducts.set(id, scannedProduct);
    return scannedProduct;
  }

  async getScannedProductsByUser(userId: number): Promise<ScannedProduct[]> {
    return Array.from(this.scannedProducts.values()).filter(
      product => product.userId === userId
    );
  }

  async getScannedProduct(id: number): Promise<ScannedProduct | undefined> {
    return this.scannedProducts.get(id);
  }

  async getIngredientByName(name: string): Promise<Ingredient | undefined> {
    return Array.from(this.ingredients.values()).find(
      ingredient => ingredient.name.toLowerCase() === name.toLowerCase() ||
                   ingredient.commonName?.toLowerCase() === name.toLowerCase()
    );
  }

  async getAllIngredients(): Promise<Ingredient[]> {
    return Array.from(this.ingredients.values());
  }

  async createIngredient(ingredient: InsertIngredient): Promise<Ingredient> {
    const id = this.currentIngredientId++;
    const newIngredient: Ingredient = { 
      ...ingredient, 
      id,
      commonName: ingredient.commonName || null,
      description: ingredient.description || null,
      allergenType: ingredient.allergenType || null,
      concerns: ingredient.concerns || []
    };
    this.ingredients.set(id, newIngredient);
    return newIngredient;
  }
}

export const storage = new MemStorage();
