import { pgTable, text, serial, integer, boolean, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  email: text("email").notNull().unique(),
  password: text("password").notNull(),
  name: text("name").notNull(),
  allergies: text("allergies").array().default([]),
  healthConditions: text("health_conditions").array().default([]),
  createdAt: timestamp("created_at").defaultNow(),
});

export const scannedProducts = pgTable("scanned_products", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  productName: text("product_name").notNull(),
  imageUrl: text("image_url"),
  extractedText: text("extracted_text"),
  ingredients: text("ingredients").array().default([]),
  analysis: jsonb("analysis"),
  safetyScore: text("safety_score").notNull(), // "safe", "moderate", "risky"
  fssaiVerified: boolean("fssai_verified").default(false),
  fssaiNumber: text("fssai_number"),
  scannedAt: timestamp("scanned_at").defaultNow(),
});

export const ingredients = pgTable("ingredients", {
  id: serial("id").primaryKey(),
  name: text("name").notNull().unique(),
  commonName: text("common_name"),
  description: text("description"),
  riskLevel: text("risk_level").notNull(), // "low", "medium", "high"
  allergenType: text("allergen_type"),
  concerns: text("concerns").array().default([]),
});

export const insertUserSchema = createInsertSchema(users).pick({
  email: true,
  password: true,
  name: true,
  allergies: true,
  healthConditions: true,
});

export const loginSchema = createInsertSchema(users).pick({
  email: true,
  password: true,
});

export const insertScannedProductSchema = createInsertSchema(scannedProducts).omit({
  id: true,
  scannedAt: true,
});

export const insertIngredientSchema = createInsertSchema(ingredients).omit({
  id: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertScannedProduct = z.infer<typeof insertScannedProductSchema>;
export type ScannedProduct = typeof scannedProducts.$inferSelect;
export type InsertIngredient = z.infer<typeof insertIngredientSchema>;
export type Ingredient = typeof ingredients.$inferSelect;
