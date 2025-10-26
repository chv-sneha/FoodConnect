#!/usr/bin/env python3
"""
Real FSSAI Validation System
- FSSAI License Database Integration
- Real-time verification
- Comprehensive validation
"""

import requests
import json
import re
from typing import Dict, Optional, List
from datetime import datetime
import sqlite3
import os

class FSSAIValidator:
    def __init__(self):
        self.fssai_api_base = "https://foscos.fssai.gov.in/api"  # Official FSSAI API
        self.backup_db_path = os.path.join(os.path.dirname(__file__), 'fssai_cache.db')
        self.init_cache_db()
        
        # FSSAI License patterns
        self.license_patterns = [
            r'fssai[:\s#-]*(\d{14})',
            r'lic[:\s#-]*no[:\s#-]*(\d{14})',
            r'license[:\s#-]*(?:no[:\s#-]*)?(\d{14})',
            r'food\s*safety[:\s#-]*(\d{14})',
            r'(\d{14})'  # Any 14-digit number
        ]
        
        # State codes for FSSAI validation
        self.state_codes = {
            '10': 'Central License',
            '11': 'Andhra Pradesh', '12': 'Arunachal Pradesh', '13': 'Assam',
            '14': 'Bihar', '15': 'Chhattisgarh', '16': 'Goa', '17': 'Gujarat',
            '18': 'Haryana', '19': 'Himachal Pradesh', '20': 'Jharkhand',
            '21': 'Karnataka', '22': 'Kerala', '23': 'Madhya Pradesh',
            '24': 'Maharashtra', '25': 'Manipur', '26': 'Meghalaya',
            '27': 'Mizoram', '28': 'Nagaland', '29': 'Odisha', '30': 'Punjab',
            '31': 'Rajasthan', '32': 'Sikkim', '33': 'Tamil Nadu',
            '34': 'Telangana', '35': 'Tripura', '36': 'Uttar Pradesh',
            '37': 'Uttarakhand', '38': 'West Bengal', '39': 'Chandigarh',
            '40': 'Delhi', '41': 'Jammu & Kashmir', '42': 'Ladakh',
            '43': 'Lakshadweep', '44': 'Puducherry', '45': 'Andaman & Nicobar'
        }
    
    def init_cache_db(self):
        """Initialize SQLite cache database for FSSAI data"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fssai_licenses (
                    license_number TEXT PRIMARY KEY,
                    business_name TEXT,
                    address TEXT,
                    state_code TEXT,
                    license_type TEXT,
                    validity_date TEXT,
                    status TEXT,
                    last_verified TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_cache (
                    license_number TEXT PRIMARY KEY,
                    is_valid BOOLEAN,
                    validation_result TEXT,
                    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache DB initialization error: {e}")
    
    def extract_fssai_numbers(self, text: str) -> List[str]:
        """Extract all possible FSSAI numbers from text"""
        found_numbers = []
        text_clean = re.sub(r'[^\w\s\d]', ' ', text)
        
        for pattern in self.license_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number = match.group(1) if match.groups() else match.group(0)
                if len(number) == 14 and number.isdigit():
                    found_numbers.append(number)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_numbers))
    
    def validate_fssai_format(self, license_number: str) -> Dict:
        """Validate FSSAI license number format"""
        if not license_number or len(license_number) != 14 or not license_number.isdigit():
            return {
                'valid_format': False,
                'error': 'FSSAI license must be exactly 14 digits'
            }
        
        # Extract components
        state_code = license_number[:2]
        business_type = license_number[2:4]
        sequence = license_number[4:12]
        check_digit = license_number[12:14]
        
        # Validate state code
        state_name = self.state_codes.get(state_code, 'Unknown State')
        
        # Validate business type (basic validation)
        business_types = {
            '01': 'Manufacturing',
            '02': 'Processing',
            '03': 'Distribution/Storage',
            '04': 'Retail',
            '05': 'Food Service',
            '06': 'Import',
            '07': 'Export',
            '08': 'Transport',
            '09': 'Club/Canteen',
            '10': 'Hawker/Vendor'
        }
        
        business_type_name = business_types.get(business_type, 'Other')
        
        return {
            'valid_format': True,
            'state_code': state_code,
            'state_name': state_name,
            'business_type': business_type,
            'business_type_name': business_type_name,
            'sequence_number': sequence,
            'check_digits': check_digit,
            'components': {
                'state': state_code,
                'business': business_type,
                'sequence': sequence,
                'check': check_digit
            }
        }
    
    def check_cache(self, license_number: str) -> Optional[Dict]:
        """Check if license is in cache"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT is_valid, validation_result, last_checked 
                FROM validation_cache 
                WHERE license_number = ? 
                AND datetime(last_checked) > datetime('now', '-7 days')
            ''', (license_number,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'cached': True,
                    'is_valid': bool(result[0]),
                    'validation_result': json.loads(result[1]),
                    'last_checked': result[2]
                }
            
            return None
        except Exception as e:
            print(f"Cache check error: {e}")
            return None
    
    def save_to_cache(self, license_number: str, is_valid: bool, validation_result: Dict):
        """Save validation result to cache"""
        try:
            conn = sqlite3.connect(self.backup_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO validation_cache 
                (license_number, is_valid, validation_result, last_checked)
                VALUES (?, ?, ?, datetime('now'))
            ''', (license_number, is_valid, json.dumps(validation_result)))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def verify_with_fssai_api(self, license_number: str) -> Dict:
        """Verify license with official FSSAI API"""
        try:
            # Official FSSAI verification endpoint
            url = f"{self.fssai_api_base}/verify-license"
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'FoodSense-AI/1.0'
            }
            
            payload = {
                'licenseNumber': license_number,
                'verificationType': 'full'
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'success':
                    license_info = data.get('licenseInfo', {})
                    
                    return {
                        'verified': True,
                        'source': 'FSSAI_API',
                        'business_name': license_info.get('businessName', 'N/A'),
                        'address': license_info.get('address', 'N/A'),
                        'license_type': license_info.get('licenseType', 'N/A'),
                        'validity_date': license_info.get('validityDate', 'N/A'),
                        'status': license_info.get('status', 'Active'),
                        'last_verified': datetime.now().isoformat()
                    }
                else:
                    return {
                        'verified': False,
                        'source': 'FSSAI_API',
                        'error': data.get('message', 'License not found'),
                        'last_verified': datetime.now().isoformat()
                    }
            
            return {
                'verified': False,
                'source': 'FSSAI_API',
                'error': f'API Error: {response.status_code}',
                'last_verified': datetime.now().isoformat()
            }
            
        except requests.exceptions.Timeout:
            return {
                'verified': False,
                'source': 'FSSAI_API',
                'error': 'API timeout - verification unavailable',
                'last_verified': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'verified': False,
                'source': 'FSSAI_API',
                'error': f'Verification failed: {str(e)}',
                'last_verified': datetime.now().isoformat()
            }
    
    def validate_license(self, license_number: str) -> Dict:
        """Complete FSSAI license validation"""
        # Step 1: Format validation
        format_result = self.validate_fssai_format(license_number)
        
        if not format_result['valid_format']:
            return {
                'license_number': license_number,
                'valid': False,
                'status': 'Invalid Format ❌',
                'message': format_result['error'],
                'details': format_result
            }
        
        # Step 2: Check cache
        cached_result = self.check_cache(license_number)
        if cached_result and cached_result['cached']:
            result = cached_result['validation_result']
            result['from_cache'] = True
            return result
        
        # Step 3: API verification
        api_result = self.verify_with_fssai_api(license_number)
        
        # Step 4: Compile final result
        final_result = {
            'license_number': license_number,
            'valid': api_result.get('verified', False),
            'status': 'Verified ✅' if api_result.get('verified') else 'Not Verified ❌',
            'message': 'Valid FSSAI license' if api_result.get('verified') else api_result.get('error', 'Verification failed'),
            'details': {
                'format_validation': format_result,
                'api_verification': api_result,
                'from_cache': False
            }
        }
        
        # Step 5: Save to cache
        self.save_to_cache(license_number, final_result['valid'], final_result)
        
        return final_result
    
    def validate_from_text(self, text: str) -> Dict:
        """Extract and validate FSSAI numbers from text"""
        extracted_numbers = self.extract_fssai_numbers(text)
        
        if not extracted_numbers:
            return {
                'found_licenses': [],
                'validation_results': [],
                'summary': {
                    'total_found': 0,
                    'valid_count': 0,
                    'invalid_count': 0,
                    'status': 'No FSSAI License Found ❌',
                    'message': 'No FSSAI license number detected in the image'
                }
            }
        
        validation_results = []
        valid_count = 0
        
        for license_number in extracted_numbers:
            result = self.validate_license(license_number)
            validation_results.append(result)
            if result['valid']:
                valid_count += 1
        
        # Determine overall status
        if valid_count > 0:
            status = f"✅ {valid_count} Valid License(s) Found"
            message = f"Found {valid_count} valid FSSAI license(s)"
        else:
            status = "❌ No Valid Licenses"
            message = f"Found {len(extracted_numbers)} license number(s) but none are valid"
        
        return {
            'found_licenses': extracted_numbers,
            'validation_results': validation_results,
            'summary': {
                'total_found': len(extracted_numbers),
                'valid_count': valid_count,
                'invalid_count': len(extracted_numbers) - valid_count,
                'status': status,
                'message': message
            }
        }
    
    def get_license_info(self, license_number: str) -> Optional[Dict]:
        """Get detailed license information"""
        validation_result = self.validate_license(license_number)
        
        if validation_result['valid']:
            api_info = validation_result['details']['api_verification']
            format_info = validation_result['details']['format_validation']
            
            return {
                'license_number': license_number,
                'business_name': api_info.get('business_name', 'N/A'),
                'address': api_info.get('address', 'N/A'),
                'state': format_info.get('state_name', 'N/A'),
                'business_type': format_info.get('business_type_name', 'N/A'),
                'license_type': api_info.get('license_type', 'N/A'),
                'validity_date': api_info.get('validity_date', 'N/A'),
                'status': api_info.get('status', 'N/A'),
                'last_verified': api_info.get('last_verified', 'N/A')
            }
        
        return None

# Test function
if __name__ == "__main__":
    validator = FSSAIValidator()
    
    # Test with sample text containing FSSAI number
    sample_text = """
    FSSAI License No: 12345678901234
    Manufactured by: ABC Food Industries
    Address: Mumbai, Maharashtra
    """
    
    result = validator.validate_from_text(sample_text)
    print("✅ FSSAI validation test:")
    print(json.dumps(result, indent=2))
    
    # Test individual license validation
    test_license = "12345678901234"
    individual_result = validator.validate_license(test_license)
    print(f"\n✅ Individual license validation for {test_license}:")
    print(json.dumps(individual_result, indent=2))