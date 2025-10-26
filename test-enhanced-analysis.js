import { EnhancedGenericAnalysis } from './server/services/enhancedGenericAnalysis.js';
import fs from 'fs';

async function testEnhancedAnalysis() {
  try {
    console.log('🧪 Testing Enhanced Generic Analysis...');
    
    // Create a test image buffer (placeholder)
    const testImagePath = './uploads/test-image.jpg';
    
    if (!fs.existsSync(testImagePath)) {
      console.log('⚠️ No test image found, creating mock buffer...');
      // Mock image buffer for testing
      const mockBuffer = Buffer.from('mock-image-data');
      
      const result = await EnhancedGenericAnalysis.quickScan(mockBuffer);
      console.log('✅ Quick scan test passed');
      console.log('Result:', JSON.stringify(result, null, 2));
      
    } else {
      const imageBuffer = fs.readFileSync(testImagePath);
      const result = await EnhancedGenericAnalysis.detailedAnalysis(imageBuffer);
      console.log('✅ Detailed analysis test passed');
      console.log('Processing time:', result.metadata?.processingTime);
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

testEnhancedAnalysis();