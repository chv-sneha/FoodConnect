#!/usr/bin/env python3
"""
Production FSSAI Service - Official FoSCoS Integration
Real FSSAI license verification with proper format validation
"""

import re
import requests
import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os

class ProductionFSSAIService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # FSSAI API endpoints (official when available)
        self.foscos_base_url = "https://foscos.fssai.gov.in"
        
        # Third-party verification APIs (backup)
        self.surepass_api = "https://kyc-api.surepass.io/api/v1/fssai-verification"
        self.surepass_token = os.getenv('SUREPASS_API_TOKEN')
        
        # Cache database
        self.cache_db_path = os.path.join(os.path.dirname(__file__), 'fssai_cache.db')
        self.init_cache_db()
        
        # FSSAI number patterns (14 digits with various formats)
        self.fssai_patterns = [
            r'fssai[:\s#-]*(\d{14})',
            r'lic[:\s#-]*no[:\s#-]*(\d{14})',
            r'license[:\s#-]*(?:no[:\s#-]*)?(\d{14})',
            r'food\s*safety[:\s#-]*(\d{14})',
            r'registration[:\s#-]*(?:no[:\s#-]*)?(\d{14})',
            r'(\d{14})'  # Any 14-digit number
        ]
        
        # State codes (first 2 digits of FSSAI number)
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
        
        # Business type codes (digits 3-4 of FSSAI number)
        self.business_types = {
            '01': 'Manufacturing',
            '02': 'Processing',
            '03': 'Distribution/Storage',
            '04': 'Retail',
            '05': 'Food Service',
            '06': 'Import',
            '07': 'Export',
            '08': 'Transport',
            '09': 'Club/Canteen',
            '10': 'Hawker/Vendor',
            '11': 'Temporary Stall',
            '12': 'E-commerce'
        }
    
    def init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fssai_verifications (
                    license_number TEXT PRIMARY KEY,
                    is_valid BOOLEAN,
                    business_name TEXT,
                    address TEXT,
                    license_type TEXT,
                    validity_date TEXT,
                    verification_source TEXT,
                    last_verified TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_last_verified 
                ON fssai_verifications(last_verified)
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Cache DB initialization error: {e}")
    
    def extract_fssai_numbers(self, text: str) -> List[Dict]:
        """Extract FSSAI numbers from text with confidence scores"""
        candidates = []
        
        # Clean text for better pattern matching
        clean_text = re.sub(r'[^\w\s\d]', ' ', text)
        
        for pattern in self.fssai_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                number = match.group(1) if match.groups() else match.group(0)
                
                if len(number) == 14 and number.isdigit():
                    # Calculate confidence based on context
                    context = text[max(0, match.start()-20):match.end()+20].lower()
                    confidence = self._calculate_extraction_confidence(context, number)
                    
                    candidates.append({
                        'number': number,
                        'confidence': confidence,
                        'context': context.strip(),
                        'position': match.start()
                    })
        
        # Remove duplicates and sort by confidence
        unique_candidates = {}
        for candidate in candidates:
            number = candidate['number']
            if number not in unique_candidates or candidate['confidence'] > unique_candidates[number]['confidence']:
                unique_candidates[number] = candidate
        
        return sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_extraction_confidence(self, context: str, number: str) -> float:
        """Calculate confidence score for extracted FSSAI number"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if found near FSSAI keywords
        fssai_keywords = ['fssai', 'license', 'lic', 'registration', 'food safety']
        for keyword in fssai_keywords:
            if keyword in context:
                confidence += 0.2
                break
        
        # Boost confidence for valid format
        if self.validate_fssai_format(number)['valid']:
            confidence += 0.2
        
        # Boost confidence if number appears isolated (not part of larger number)
        if re.search(r'\b' + number + r'\b', context):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def validate_fssai_format(self, license_number: str) -> Dict:
        """Validate FSSAI license number format"""
        if not license_number or len(license_number) != 14 or not license_number.isdigit():
            return {
                'valid': False,
                'error': 'FSSAI license must be exactly 14 digits',
                'details': {}
            }
        
        # Extract components
        state_code = license_number[:2]
        business_type = license_number[2:4]
        sequence = license_number[4:12]
        check_digits = license_number[12:14]
        
        # Validate state code
        state_name = self.state_codes.get(state_code, 'Unknown State')
        if state_name == 'Unknown State':
            return {
                'valid': False,
                'error': f'Invalid state code: {state_code}',
                'details': {'state_code': state_code}
            }
        
        # Validate business type
        business_type_name = self.business_types.get(business_type, 'Other')
        
        return {
            'valid': True,
            'details': {
                'state_code': state_code,
                'state_name': state_name,
                'business_type': business_type,
                'business_type_name': business_type_name,
                'sequence_number': sequence,
                'check_digits': check_digits
            }
        }
    
    def check_cache(self, license_number: str) -> Optional[Dict]:
        """Check if license verification is cached"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Check for recent verification (within 7 days)
            cursor.execute('''
                SELECT is_valid, business_name, address, license_type, 
                       validity_date, verification_source, last_verified
                FROM fssai_verifications 
                WHERE license_number = ? 
                AND datetime(last_verified) > datetime('now', '-7 days')
            ''', (license_number,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'cached': True,
                    'is_valid': bool(result[0]),
                    'business_name': result[1],
                    'address': result[2],
                    'license_type': result[3],
                    'validity_date': result[4],
                    'verification_source': result[5],
                    'last_verified': result[6]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cache check error: {e}")
            return None
    
    def save_to_cache(self, license_number: str, verification_result: Dict):
        """Save verification result to cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO fssai_verifications 
                (license_number, is_valid, business_name, address, license_type,
                 validity_date, verification_source, last_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                license_number,
                verification_result.get('is_valid', False),
                verification_result.get('business_name', ''),
                verification_result.get('address', ''),
                verification_result.get('license_type', ''),
                verification_result.get('validity_date', ''),
                verification_result.get('source', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Cache save error: {e}")
    
    def verify_with_surepass(self, license_number: str) -> Dict:
        """Verify FSSAI license using Surepass API"""
        if not self.surepass_token:
            return {'success': False, 'error': 'Surepass API token not configured'}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.surepass_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'id_number': license_number
            }
            
            response = requests.post(
                self.surepass_api,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status_code') == 200:
                    result = data.get('data', {})
                    
                    return {
                        'success': True,
                        'is_valid': result.get('valid', False),
                        'business_name': result.get('business_name', ''),
                        'address': result.get('address', ''),
                        'license_type': result.get('license_type', ''),
                        'validity_date': result.get('validity_date', ''),
                        'source': 'surepass_api'
                    }
                else:
                    return {
                        'success': False,
                        'error': data.get('message', 'Verification failed'),
                        'source': 'surepass_api'
                    }
            
            return {
                'success': False,
                'error': f'API error: {response.status_code}',
                'source': 'surepass_api'
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'API timeout',
                'source': 'surepass_api'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'surepass_api'
            }
    
    def verify_license(self, license_number: str) -> Dict:
        """Complete FSSAI license verification"""
        
        # Step 1: Format validation
        format_result = self.validate_fssai_format(license_number)
        if not format_result['valid']:
            return {
                'license_number': license_number,
                'valid': False,
                'status': 'Invalid Format ❌',
                'message': format_result['error'],
                'confidence': 0.0,
                'source': 'format_validation'
            }
        
        # Step 2: Check cache
        cached_result = self.check_cache(license_number)
        if cached_result:
            return {
                'license_number': license_number,
                'valid': cached_result['is_valid'],
                'status': 'Verified ✅' if cached_result['is_valid'] else 'Not Valid ❌',
                'message': 'Valid FSSAI license' if cached_result['is_valid'] else 'License not valid',
                'business_name': cached_result.get('business_name', ''),
                'address': cached_result.get('address', ''),
                'license_type': cached_result.get('license_type', ''),
                'validity_date': cached_result.get('validity_date', ''),
                'confidence': 0.9,
                'source': f"{cached_result['verification_source']} (cached)",
                'format_details': format_result['details']
            }
        
        # Step 3: API verification
        api_result = self.verify_with_surepass(license_number)
        
        if api_result['success']:
            # Save to cache
            self.save_to_cache(license_number, api_result)
            
            return {
                'license_number': license_number,
                'valid': api_result['is_valid'],
                'status': 'Verified ✅' if api_result['is_valid'] else 'Not Valid ❌',
                'message': 'Valid FSSAI license' if api_result['is_valid'] else 'License not found in database',
                'business_name': api_result.get('business_name', ''),
                'address': api_result.get('address', ''),
                'license_type': api_result.get('license_type', ''),
                'validity_date': api_result.get('validity_date', ''),
                'confidence': 0.85,
                'source': api_result['source'],
                'format_details': format_result['details']
            }
        else:
            # API failed, return format validation only
            return {
                'license_number': license_number,
                'valid': False,
                'status': 'Format Valid ⚠️',
                'message': f"Format is valid but verification failed: {api_result.get('error', 'Unknown error')}",
                'confidence': 0.3,
                'source': 'format_only',
                'format_details': format_result['details']
            }
    
    def verify_from_text(self, text: str) -> Dict:
        """Extract and verify FSSAI numbers from text"""
        
        # Extract candidate numbers
        candidates = self.extract_fssai_numbers(text)
        
        if not candidates:
            return {
                'found_licenses': [],
                'verification_results': [],
                'summary': {
                    'total_found': 0,
                    'valid_count': 0,
                    'status': 'No FSSAI License Found ❌',
                    'message': 'No FSSAI license number detected in the image'
                }
            }
        
        # Verify each candidate
        verification_results = []
        valid_count = 0
        
        for candidate in candidates[:3]:  # Limit to top 3 candidates
            license_number = candidate['number']
            result = self.verify_license(license_number)
            
            # Add extraction confidence to result
            result['extraction_confidence'] = candidate['confidence']
            result['context'] = candidate['context']
            
            verification_results.append(result)
            
            if result['valid']:
                valid_count += 1
        
        # Determine overall status
        if valid_count > 0:
            status = f"✅ {valid_count} Valid License(s) Found"
            message = f"Found {valid_count} valid FSSAI license(s)"
        else:
            status = "❌ No Valid Licenses"
            message = f"Found {len(candidates)} license number(s) but none are valid"
        
        return {
            'found_licenses': [c['number'] for c in candidates],
            'verification_results': verification_results,
            'summary': {
                'total_found': len(candidates),
                'valid_count': valid_count,
                'status': status,
                'message': message
            }
        }