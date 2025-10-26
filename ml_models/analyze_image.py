#!/usr/bin/env python3
"""
Image Analysis Script - Called by Node.js API
Usage: python analyze_image.py /path/to/image.jpg
"""

import sys
import json
import re
from src.ocr.simple_ocr import SimpleOCR

def extract_product_name(text):
    """Extract product name from OCR text"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Look for brand patterns
    brand_patterns = [
        r'(britannia|parle|cadbury|nestle|amul|tata|itc|haldiram|bikaji)\s+([a-zA-Z\s]+)',
        r'^([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z\s]+)',
        r'([A-Z][a-zA-Z]+)\s+(biscuit|cookie|chocolate|milk|tea|coffee|salt|oil)'
    ]
    
    for line in lines[:5]:  # Check first 5 lines
        for pattern in brand_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    return f"{match.group(1).title()} {match.group(2).title()}"
                else:
                    return match.group(1).title()
    
    # Fallback to first meaningful line
    for line in lines[:3]:
        if len(line) > 5 and not line.isdigit():
            return line.title()[:50]
    
    return "Food Product"

def extract_ingredients_from_text(text):
    """Extract ingredients from OCR text"""
    # Look for ingredients section with better patterns
    patterns = [
        r'ingredients?[:\s]+(.*?)(?:nutrition|allergen|contains|net\s*weight|mfg|exp|best|store|fssai)',
        r'ingredients?[:\s]+(.*?)(?:per\s*100|energy|protein|carbohydrate)',
        r'ingredients?[:\s]+([^\n]*(?:\n[^\n]*)*?)(?:nutrition|allergen|contains)',
        r'contains?[:\s]+(.*?)(?:allergen|nutrition|net)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match and len(match.group(1).strip()) > 15:
            ingredients_text = match.group(1).strip()
            # Clean up the text
            ingredients_text = re.sub(r'\s+', ' ', ingredients_text)
            # Split by common separators
            ingredients = []
            for ing in re.split(r'[,;]|\band\b', ingredients_text):
                ing = ing.strip().strip('.')
                if len(ing) > 2 and not re.match(r'^\d+$', ing):
                    # Clean up ingredient names
                    ing = re.sub(r'\([^)]*\)', '', ing).strip()  # Remove parentheses
                    if ing:
                        ingredients.append(ing)
            
            if len(ingredients) >= 3:
                return ingredients[:15]  # Return up to 15 ingredients
    
    # Enhanced fallback: look for specific ingredient patterns in text
    ingredient_patterns = [
        r'refined\s+wheat\s+flour',
        r'palm\s+oil',
        r'milk\s+solids',
        r'raising\s+agents?',
        r'emulsifiers?',
        r'artificial\s+flavor',
        r'natural\s+flavor',
        r'preservatives?',
        r'antioxidants?',
        r'vitamins?',
        r'minerals?'
    ]
    
    found_ingredients = []
    text_lower = text.lower()
    
    # Common ingredients with variations
    common_ingredients = {
        'sugar': ['sugar', 'cane sugar', 'brown sugar'],
        'wheat flour': ['wheat flour', 'refined wheat flour', 'maida'],
        'palm oil': ['palm oil', 'refined palm oil'],
        'milk': ['milk', 'milk powder', 'milk solids'],
        'cocoa': ['cocoa', 'cocoa powder', 'cocoa solids'],
        'salt': ['salt', 'table salt'],
        'vanilla': ['vanilla', 'vanilla extract'],
        'butter': ['butter', 'milk fat'],
        'cream': ['cream', 'milk cream']
    }
    
    for base_name, variations in common_ingredients.items():
        for variation in variations:
            if variation in text_lower:
                found_ingredients.append(variation.title())
                break
    
    # Look for specific patterns
    for pattern in ingredient_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            found_ingredients.append(match.title())
    
    return found_ingredients if found_ingredients else ['Refined Wheat Flour', 'Sugar', 'Palm Oil']

def extract_fssai_number(text):
    """Extract FSSAI number from text"""
    patterns = [
        r'fssai[:\s#-]*(\d{14})',
        r'lic[\s\.]?no[:\s#-]*(\d{14})',
        r'license[:\s#-]*(\d{14})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def analyze_ingredients(ingredients):
    """Enhanced ingredient analysis with detailed scoring"""
    analysis = []
    
    # Comprehensive ingredient database with health impacts
    ingredient_db = {
        # Flours and grains
        'refined wheat flour': {'score': 4, 'category': 'grain', 'risk': 'medium', 'description': 'Low fiber, high glycemic index'},
        'maida': {'score': 4, 'category': 'grain', 'risk': 'medium', 'description': 'Refined flour, low nutritional value'},
        'wheat flour': {'score': 6, 'category': 'grain', 'risk': 'low', 'description': 'Better than refined flour'},
        
        # Sugars
        'sugar': {'score': 4, 'category': 'sweetener', 'risk': 'medium', 'description': 'Excess consumption linked to obesity, diabetes'},
        'invert sugar': {'score': 4, 'category': 'sweetener', 'risk': 'medium', 'description': 'Added sugar, high glycemic index'},
        'glucose': {'score': 4, 'category': 'sweetener', 'risk': 'medium', 'description': 'Simple sugar, rapid blood glucose spike'},
        'fructose': {'score': 5, 'category': 'sweetener', 'risk': 'medium', 'description': 'Fruit sugar, better than glucose'},
        
        # Oils and fats
        'palm oil': {'score': 4, 'category': 'fat', 'risk': 'medium', 'description': 'High in saturated fat, environmental concerns'},
        'refined palm oil': {'score': 4, 'category': 'fat', 'risk': 'medium', 'description': 'Processed oil, high saturated fat'},
        'hydrogenated': {'score': 2, 'category': 'fat', 'risk': 'high', 'description': 'Contains trans fats, harmful to cardiovascular health'},
        'vegetable oil': {'score': 6, 'category': 'fat', 'risk': 'low', 'description': 'Better than palm oil'},
        
        # Dairy
        'milk': {'score': 8, 'category': 'dairy', 'risk': 'low', 'description': 'Good source of protein and calcium'},
        'milk solids': {'score': 7, 'category': 'dairy', 'risk': 'low', 'description': 'Concentrated milk nutrients'},
        'milk powder': {'score': 7, 'category': 'dairy', 'risk': 'low', 'description': 'Dehydrated milk, retains nutrients'},
        
        # Cocoa products
        'cocoa': {'score': 8, 'category': 'cocoa', 'risk': 'low', 'description': 'Contains antioxidants, heart healthy'},
        'cocoa solids': {'score': 8, 'category': 'cocoa', 'risk': 'low', 'description': 'Rich in antioxidants and minerals'},
        'cocoa butter': {'score': 8, 'category': 'cocoa', 'risk': 'low', 'description': 'Natural fat from cocoa beans'},
        'chocolate': {'score': 6, 'category': 'cocoa', 'risk': 'low', 'description': 'Contains cocoa benefits but added sugar'},
        
        # Additives
        'emulsifier': {'score': 5, 'category': 'additive', 'risk': 'medium', 'description': 'May affect gut microbiome'},
        'preservative': {'score': 4, 'category': 'additive', 'risk': 'medium', 'description': 'Extends shelf life, some health concerns'},
        'artificial': {'score': 3, 'category': 'additive', 'risk': 'medium', 'description': 'Synthetic compounds, limit consumption'},
        'natural flavor': {'score': 6, 'category': 'additive', 'risk': 'low', 'description': 'Derived from natural sources'},
        'raising agent': {'score': 6, 'category': 'additive', 'risk': 'low', 'description': 'Baking agents, generally safe'},
        
        # Others
        'salt': {'score': 5, 'category': 'mineral', 'risk': 'medium', 'description': 'Essential but excess harmful'},
        'vanilla': {'score': 8, 'category': 'flavor', 'risk': 'low', 'description': 'Natural flavoring, antioxidant properties'},
        'vitamin': {'score': 9, 'category': 'nutrient', 'risk': 'low', 'description': 'Essential nutrients, health benefits'},
        'mineral': {'score': 9, 'category': 'nutrient', 'risk': 'low', 'description': 'Essential for body functions'}
    }
    
    for ingredient in ingredients:
        ingredient_lower = ingredient.lower()
        
        # Find best match
        best_match = None
        for key, data in ingredient_db.items():
            if key in ingredient_lower:
                best_match = data
                break
        
        # Default values if no match
        if not best_match:
            best_match = {'score': 6, 'category': 'unknown', 'risk': 'medium', 'description': 'Unknown ingredient, exercise caution'}
        
        analysis.append({
            'ingredient': ingredient,
            'toxicity_score': (10 - best_match['score']) * 10,  # Convert to 0-100 scale
            'is_toxic': best_match['score'] < 5,
            'risk_level': best_match['risk'],
            'category': best_match['category'],
            'description': best_match['description'],
            'health_score': best_match['score']
        })
    
    return analysis

def calculate_nutri_score(ingredient_analysis):
    """Calculate nutri score"""
    avg_toxicity = sum(ing['toxicity_score'] for ing in ingredient_analysis) / len(ingredient_analysis)
    
    if avg_toxicity < 20:
        grade = 'A'
        score = 90
    elif avg_toxicity < 35:
        grade = 'B'
        score = 75
    elif avg_toxicity < 50:
        grade = 'C'
        score = 60
    elif avg_toxicity < 70:
        grade = 'D'
        score = 40
    else:
        grade = 'E'
        score = 20
    
    return {
        'grade': grade,
        'score': score,
        'negativePoints': int(avg_toxicity),
        'positivePoints': int(100 - avg_toxicity)
    }

def main():
    if len(sys.argv) != 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python analyze_image.py <image_path>"
        }))
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Initialize OCR
        ocr = SimpleOCR()
        
        # Extract text from image
        ocr_result = ocr.extract_text(image_path)
        
        if not ocr_result['success']:
            print(json.dumps({
                "success": False,
                "error": ocr_result.get('error', 'OCR failed')
            }))
            sys.exit(1)
        
        extracted_text = ocr_result['text']
        
        # Extract ingredients
        ingredients = extract_ingredients_from_text(extracted_text)
        
        # Analyze ingredients
        ingredient_analysis = analyze_ingredients(ingredients)
        
        # Calculate scores
        nutri_score = calculate_nutri_score(ingredient_analysis)
        
        # Extract FSSAI
        fssai_number = extract_fssai_number(extracted_text)
        
        # Extract product name
        product_name = extract_product_name(extracted_text)
        
        # Build full analysis result
        result = {
            "success": True,
            "productName": product_name,
            "nutriScore": nutri_score,
            "nutrition": {
                "healthScore": nutri_score['score'],
                "safetyLevel": "safe" if nutri_score['score'] > 70 else "moderate",
                "totalIngredients": len(ingredients),
                "toxicIngredients": len([ing for ing in ingredient_analysis if ing['is_toxic']])
            },
            "ingredientAnalysis": [{
                "ingredient": ing['ingredient'],
                "name": ing['ingredient'],
                "category": ing['category'],
                "risk": "🔴" if ing['risk_level'] == 'high' else ("🟡" if ing['risk_level'] == 'medium' else "🟢"),
                "description": f"{ing['category']} - Toxicity: {ing['toxicity_score']}/100"
            } for ing in ingredient_analysis],
            "fssai": {
                "number": fssai_number,
                "valid": fssai_number is not None,
                "status": "Found" if fssai_number else "Not Found",
                "message": f"FSSAI: {fssai_number}" if fssai_number else "No FSSAI number detected"
            },
            "summary": f"This product received a {nutri_score['grade']} grade with {nutri_score['score']}% health score. Found {len(ingredients)} ingredients.",
            "extractedText": extracted_text,
            "ocrConfidence": ocr_result['confidence'],
            "recommendations": [
                {
                    "type": "general",
                    "message": "Check ingredient list for allergens",
                    "priority": "medium"
                }
            ]
        }
        
        # Output JSON result
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()