#!/usr/bin/env python3
"""
Demo: Enhanced Food Analysis System
Shows complete analysis pipeline with sample data
"""

import sys
import json
import os
from datetime import datetime

# Add paths
sys.path.append('ml_models')
sys.path.append('ml_models/src')

def demo_complete_analysis():
    """Demonstrate complete food analysis pipeline"""
    
    print("🚀 FoodSense AI - Complete Analysis Demo")
    print("=" * 60)
    
    # Sample food label text (simulated OCR result)
    sample_ocr_text = """
    BRITANNIA GOOD DAY BUTTER COOKIES
    
    Ingredients: Wheat Flour, Sugar, Edible Vegetable Oil (Palm Oil), 
    Butter (5%), Milk Solids, Salt, Raising Agents (Sodium Bicarbonate, 
    Ammonium Bicarbonate), Emulsifiers (Soy Lecithin), Artificial 
    Vanilla Flavoring
    
    Nutrition Information (Per 100g):
    Energy: 456 kcal
    Protein: 6.8g
    Carbohydrates: 70.2g
    Total Fat: 16.2g
    Trans Fat: 0.1g
    Saturated Fat: 8.1g
    Sugar: 18.5g
    Dietary Fiber: 2.1g
    Sodium: 312mg
    
    FSSAI License No: 10017047000694
    Mfd by: Britannia Industries Ltd
    """
    
    print("📄 Sample Food Label Text:")
    print("-" * 40)
    print(sample_ocr_text.strip())
    print("-" * 40)
    
    try:
        # Import components
        from ml_models.src.nutrition.nutrition_analyzer import NutritionAnalyzer
        from ml_models.src.fssai.fssai_validator import FSSAIValidator
        from enhanced_analyze_image import EnhancedFoodAnalyzer
        
        # Initialize analyzers
        nutrition_analyzer = NutritionAnalyzer()
        fssai_validator = FSSAIValidator()
        enhanced_analyzer = EnhancedFoodAnalyzer()
        
        print("\n🔍 STEP 1: Extract Product Information")
        print("-" * 40)
        
        # Extract product name
        product_name = enhanced_analyzer.extract_product_name(sample_ocr_text)
        print(f"📦 Product Name: {product_name}")
        
        # Extract ingredients
        ingredients = enhanced_analyzer.extract_ingredients_from_text(sample_ocr_text)
        print(f"📋 Ingredients Found: {len(ingredients)}")
        for i, ingredient in enumerate(ingredients[:5], 1):
            print(f"   {i}. {ingredient}")
        if len(ingredients) > 5:
            print(f"   ... and {len(ingredients) - 5} more")
        
        print("\n🥗 STEP 2: Nutrition Analysis")
        print("-" * 40)
        
        # Extract nutrition from text
        nutrition_result = nutrition_analyzer.extract_nutrition_from_text(sample_ocr_text)
        nutrition_facts = nutrition_result['nutrition_facts']
        
        print("📊 Extracted Nutrition Facts:")
        for nutrient, value in nutrition_facts.items():
            if value > 0:
                unit = 'kcal' if nutrient == 'calories' else 'g' if nutrient != 'sodium' else 'mg'
                print(f"   • {nutrient.title()}: {value}{unit}")
        
        # Calculate Nutri-Score
        nutri_score = nutrition_analyzer.calculate_nutri_score(nutrition_facts)
        print(f"\n🏆 Nutri-Score: {nutri_score['grade']} (Score: {nutri_score['score']}%)")
        print(f"   • Negative Points: {nutri_score['negative_points']}")
        print(f"   • Positive Points: {nutri_score['positive_points']}")
        
        print("\n⚠️ STEP 3: Ingredient Risk Analysis")
        print("-" * 40)
        
        # Analyze ingredient risks
        risk_database = {
            'trans fat': {'risk': 'high', 'toxicity': 95, 'category': 'harmful_fat'},
            'palm oil': {'risk': 'medium', 'toxicity': 55, 'category': 'fat'},
            'sugar': {'risk': 'medium', 'toxicity': 65, 'category': 'sweetener'},
            'sodium bicarbonate': {'risk': 'low', 'toxicity': 25, 'category': 'raising_agent'},
            'soy lecithin': {'risk': 'low', 'toxicity': 20, 'category': 'emulsifier'},
            'artificial vanilla': {'risk': 'medium', 'toxicity': 45, 'category': 'flavoring'},
            'wheat flour': {'risk': 'safe', 'toxicity': 10, 'category': 'grain'},
            'butter': {'risk': 'low', 'toxicity': 30, 'category': 'dairy'},
            'milk solids': {'risk': 'safe', 'toxicity': 15, 'category': 'dairy'}
        }
        
        high_risk_ingredients = []
        total_toxicity = 0
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            risk_data = None
            
            for risk_ingredient, data in risk_database.items():
                if risk_ingredient in ingredient_lower:
                    risk_data = data
                    break
            
            if not risk_data:
                risk_data = {'risk': 'low', 'toxicity': 30, 'category': 'unknown'}
            
            total_toxicity += risk_data['toxicity']
            
            risk_emoji = {'safe': '🟢', 'low': '🟡', 'medium': '🟠', 'high': '🔴'}.get(risk_data['risk'], '🟡')
            
            print(f"   {risk_emoji} {ingredient} - {risk_data['category']} (Toxicity: {risk_data['toxicity']}/100)")
            
            if risk_data['risk'] in ['high', 'medium'] and risk_data['toxicity'] > 50:
                high_risk_ingredients.append(ingredient)
        
        avg_toxicity = total_toxicity / len(ingredients) if ingredients else 0
        health_score = max(0, 100 - avg_toxicity)
        
        print(f"\n📈 Overall Health Score: {health_score:.1f}%")
        print(f"📊 Average Toxicity: {avg_toxicity:.1f}/100")
        
        print("\n🛡️ STEP 4: FSSAI Verification")
        print("-" * 40)
        
        # FSSAI validation
        fssai_result = fssai_validator.validate_from_text(sample_ocr_text)
        
        print(f"🔍 FSSAI Status: {fssai_result['summary']['status']}")
        print(f"📄 Message: {fssai_result['summary']['message']}")
        
        if fssai_result['summary']['total_found'] > 0:
            for validation in fssai_result['validation_results']:
                license_num = validation['license_number']
                format_valid = validation['details']['format_validation']
                print(f"   • License: {license_num}")
                print(f"   • State: {format_valid.get('state_name', 'Unknown')}")
                print(f"   • Business Type: {format_valid.get('business_type_name', 'Unknown')}")
        
        print("\n💡 STEP 5: Health Recommendations")
        print("-" * 40)
        
        recommendations = []
        
        if high_risk_ingredients:
            recommendations.append({
                'type': 'warning',
                'message': f'Contains {len(high_risk_ingredients)} concerning ingredients: {", ".join(high_risk_ingredients[:3])}',
                'priority': 'high'
            })
        
        if nutrition_facts.get('sugar', 0) > 15:
            recommendations.append({
                'type': 'health_alert',
                'message': 'High sugar content - limit consumption if diabetic',
                'priority': 'medium'
            })
        
        if nutrition_facts.get('sodium', 0) > 300:
            recommendations.append({
                'type': 'health_alert',
                'message': 'High sodium content - monitor if you have hypertension',
                'priority': 'medium'
            })
        
        if avg_toxicity > 50:
            recommendations.append({
                'type': 'substitute',
                'message': 'Consider healthier alternatives with fewer processed ingredients',
                'priority': 'medium'
            })
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {'high': '🚨', 'medium': '⚠️', 'low': 'ℹ️'}.get(rec['priority'], 'ℹ️')
                print(f"   {priority_emoji} {rec['message']}")
        else:
            print("   ✅ No major health concerns detected")
        
        print("\n📋 STEP 6: Final Summary")
        print("-" * 40)
        
        if avg_toxicity < 25: safety_level = 'excellent'
        elif avg_toxicity < 40: safety_level = 'good'
        elif avg_toxicity < 55: safety_level = 'moderate'
        elif avg_toxicity < 70: safety_level = 'poor'
        else: safety_level = 'very_poor'
        
        summary = f"""
🏷️ Product: {product_name}
🏆 Nutri-Score: {nutri_score['grade']} ({nutri_score['score']}%)
💚 Health Score: {health_score:.1f}%
🛡️ Safety Level: {safety_level.title()}
📊 Ingredients: {len(ingredients)} total, {len(high_risk_ingredients)} concerning
🔍 FSSAI: {fssai_result['summary']['status']}
⚠️ Recommendations: {len(recommendations)} alerts
        """.strip()
        
        print(summary)
        
        print("\n" + "=" * 60)
        print("✅ Demo Complete! This is what users will see when they upload food labels.")
        print("🚀 Start the server with 'npm run dev' and try it yourself!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_complete_analysis()