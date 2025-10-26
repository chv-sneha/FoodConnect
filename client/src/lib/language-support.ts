// Multi-language support for regional languages
export interface LanguageSupport {
  code: string;
  name: string;
  nativeName: string;
  enabled: boolean;
}

export const SUPPORTED_LANGUAGES: LanguageSupport[] = [
  { code: 'en', name: 'English', nativeName: 'English', enabled: true },
  { code: 'hi', name: 'Hindi', nativeName: 'हिंदी', enabled: true },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்', enabled: true },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা', enabled: true },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు', enabled: true },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी', enabled: true },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી', enabled: true },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ', enabled: true },
];

// Translation database for common food safety terms
const TRANSLATIONS: Record<string, Record<string, string>> = {
  en: {
    // Safety levels
    'safe': 'Safe',
    'moderate': 'Moderate Risk',
    'risky': 'High Risk',
    'dangerous': 'Dangerous',
    
    // Common ingredients
    'sugar': 'Sugar',
    'salt': 'Salt',
    'oil': 'Oil',
    'milk': 'Milk',
    'wheat': 'Wheat',
    
    // Analysis results
    'ingredientAnalysis': 'Ingredient Analysis',
    'allergenWarning': 'Allergen Warning',
    'toxicityScore': 'Toxicity Score',
    'recommendations': 'Recommendations',
    'safetyRating': 'Safety Rating',
    
    // Actions
    'scanProduct': 'Scan Product',
    'viewResults': 'View Results',
    'findAlternatives': 'Find Alternatives',
  },
  
  hi: {
    // Safety levels
    'safe': 'सुरक्षित',
    'moderate': 'मध्यम जोखिम',
    'risky': 'उच्च जोखिम',
    'dangerous': 'खतरनाक',
    
    // Common ingredients
    'sugar': 'चीनी',
    'salt': 'नमक',
    'oil': 'तेल',
    'milk': 'दूध',
    'wheat': 'गेहूं',
    
    // Analysis results
    'ingredientAnalysis': 'सामग्री विश्लेषण',
    'allergenWarning': 'एलर्जी चेतावनी',
    'toxicityScore': 'विषाक्तता स्कोर',
    'recommendations': 'सिफारिशें',
    'safetyRating': 'सुरक्षा रेटिंग',
    
    // Actions
    'scanProduct': 'उत्पाद स्कैन करें',
    'viewResults': 'परिणाम देखें',
    'findAlternatives': 'विकल्प खोजें',
  },
  
  ta: {
    // Safety levels
    'safe': 'பாதுகாப்பு',
    'moderate': 'நடுத்தர ஆபத்து',
    'risky': 'அதிக ஆபத்து',
    'dangerous': 'ஆபத்தானது',
    
    // Common ingredients
    'sugar': 'சர்க்கரை',
    'salt': 'உப்பு',
    'oil': 'எண்ணெய்',
    'milk': 'பால்',
    'wheat': 'கோதுமை',
    
    // Analysis results
    'ingredientAnalysis': 'பொருள் பகுப்பாய்வு',
    'allergenWarning': 'ஒவ்வாமை எச்சரிக்கை',
    'toxicityScore': 'நச்சுத்தன்மை மதிப்பெண்',
    'recommendations': 'பரிந்துரைகள்',
    'safetyRating': 'பாதுகாப்பு மதிப்பீடு',
    
    // Actions
    'scanProduct': 'தயாரிப்பு ஸ்கேன்',
    'viewResults': 'முடிவுகளைப் பார்க்க',
    'findAlternatives': 'மாற்றுகளைக் கண்டறியவும்',
  },
  
  bn: {
    // Safety levels
    'safe': 'নিরাপদ',
    'moderate': 'মাঝারি ঝুঁকি',
    'risky': 'উচ্চ ঝুঁকি',
    'dangerous': 'বিপজ্জনক',
    
    // Common ingredients
    'sugar': 'চিনি',
    'salt': 'লবণ',
    'oil': 'তেল',
    'milk': 'দুধ',
    'wheat': 'গম',
    
    // Analysis results
    'ingredientAnalysis': 'উপাদান বিশ্লেষণ',
    'allergenWarning': 'অ্যালার্জি সতর্কতা',
    'toxicityScore': 'বিষাক্ততার স্কোর',
    'recommendations': 'সুপারিশ',
    'safetyRating': 'নিরাপত্তা রেটিং',
    
    // Actions
    'scanProduct': 'পণ্য স্ক্যান করুন',
    'viewResults': 'ফলাফল দেখুন',
    'findAlternatives': 'বিকল্প খুঁজুন',
  }
};

export function translateText(key: string, languageCode: string = 'en'): string {
  const translations = TRANSLATIONS[languageCode];
  if (!translations) return key;
  return translations[key] || key;
}

export function translateAnalysisReport(analysis: any, languageCode: string): any {
  if (languageCode === 'en') return analysis;
  
  // This would typically call a translation API with proper authentication
  // For now, we'll translate key fields using our basic dictionary
  return {
    ...analysis,
    safetyRating: translateText(analysis.safetyScore?.toLowerCase() || 'safe', languageCode),
    // Add more translations as needed
  };
}

// Voice synthesis for non-literate users
export function speakAnalysisResults(analysis: any, languageCode: string = 'en'): void {
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis not supported');
    return;
  }

  const translated = translateAnalysisReport(analysis, languageCode);
  
  let textToSpeak = '';
  
  // Safety rating
  textToSpeak += `${translateText('safetyRating', languageCode)}: ${translated.safetyRating}. `;
  
  // Allergen warnings
  if (analysis.personalizedWarnings && analysis.personalizedWarnings.length > 0) {
    textToSpeak += `${translateText('allergenWarning', languageCode)}: `;
    textToSpeak += analysis.personalizedWarnings.join(', ') + '. ';
  }
  
  // Recommendations
  if (analysis.recommendations && analysis.recommendations.length > 0) {
    textToSpeak += `${translateText('recommendations', languageCode)}: `;
    textToSpeak += analysis.recommendations.slice(0, 2).join('. ') + '. ';
  }

  const utterance = new SpeechSynthesisUtterance(textToSpeak);
  
  // Set language
  const languageMap: Record<string, string> = {
    'en': 'en-US',
    'hi': 'hi-IN',
    'ta': 'ta-IN',
    'bn': 'bn-IN',
    'te': 'te-IN',
    'mr': 'mr-IN',
    'gu': 'gu-IN',
    'kn': 'kn-IN',
  };
  
  utterance.lang = languageMap[languageCode] || 'en-US';
  utterance.rate = 0.8; // Slower for better comprehension
  utterance.pitch = 1.0;
  
  speechSynthesis.speak(utterance);
}

export function getAvailableVoices(languageCode: string): SpeechSynthesisVoice[] {
  if (!('speechSynthesis' in window)) return [];
  
  const voices = speechSynthesis.getVoices();
  const languageMap: Record<string, string[]> = {
    'en': ['en-US', 'en-IN'],
    'hi': ['hi-IN'],
    'ta': ['ta-IN'],
    'bn': ['bn-IN'],
    'te': ['te-IN'],
    'mr': ['mr-IN'],
    'gu': ['gu-IN'],
    'kn': ['kn-IN'],
  };
  
  const targetLangs = languageMap[languageCode] || ['en-US'];
  return voices.filter(voice => 
    targetLangs.some(lang => voice.lang.startsWith(lang))
  );
}