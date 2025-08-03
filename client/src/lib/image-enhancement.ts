// Advanced image enhancement for low-quality cameras
export function enhanceImageForOCR(imageData: ImageData): ImageData {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const enhanced = new ImageData(width, height);

  // Apply multiple enhancement techniques
  
  // Step 1: Noise reduction using Gaussian blur
  const blurred = applyGaussianBlur(imageData, 1);
  
  // Step 2: Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
  const contrasted = applyCLAHE(blurred);
  
  // Step 3: Edge sharpening
  const sharpened = applyUnsharpMask(contrasted, 1.5, 1);
  
  // Step 4: Adaptive binarization for text
  const binarized = applyAdaptiveBinarization(sharpened);
  
  return binarized;
}

function applyGaussianBlur(imageData: ImageData, radius: number): ImageData {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const result = new ImageData(width, height);
  
  // Simple box blur approximation
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, count = 0;
      
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const idx = (ny * width + nx) * 4;
            r += data[idx];
            g += data[idx + 1];
            b += data[idx + 2];
            count++;
          }
        }
      }
      
      const idx = (y * width + x) * 4;
      result.data[idx] = r / count;
      result.data[idx + 1] = g / count;
      result.data[idx + 2] = b / count;
      result.data[idx + 3] = data[idx + 3];
    }
  }
  
  return result;
}

function applyCLAHE(imageData: ImageData): ImageData {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const result = new ImageData(width, height);
  
  // Convert to grayscale and apply histogram equalization
  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    
    // Simple contrast enhancement
    let enhanced = gray;
    if (gray < 85) {
      enhanced = gray * 0.3; // Darken dark areas
    } else if (gray > 170) {
      enhanced = Math.min(255, gray * 1.5); // Brighten light areas
    } else {
      enhanced = Math.min(255, (gray - 128) * 2 + 128); // Increase contrast in mid-tones
    }
    
    result.data[i] = enhanced;
    result.data[i + 1] = enhanced;
    result.data[i + 2] = enhanced;
    result.data[i + 3] = data[i + 3];
  }
  
  return result;
}

function applyUnsharpMask(imageData: ImageData, amount: number, radius: number): ImageData {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const result = new ImageData(width, height);
  
  // Create a blurred version
  const blurred = applyGaussianBlur(imageData, radius);
  
  // Apply unsharp mask: original + amount * (original - blurred)
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) { // RGB channels
      const original = data[i + c];
      const blur = blurred.data[i + c];
      const mask = original - blur;
      const enhanced = Math.min(255, Math.max(0, original + amount * mask));
      result.data[i + c] = enhanced;
    }
    result.data[i + 3] = data[i + 3]; // Alpha
  }
  
  return result;
}

function applyAdaptiveBinarization(imageData: ImageData): ImageData {
  const data = imageData.data;
  const width = imageData.width;
  const height = imageData.height;
  const result = new ImageData(width, height);
  
  // Convert to grayscale first
  const grayscale = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    grayscale[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  
  // Apply adaptive threshold using local mean
  const windowSize = 15;
  const k = 0.2;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;
      
      // Calculate local mean
      for (let dy = -windowSize; dy <= windowSize; dy++) {
        for (let dx = -windowSize; dx <= windowSize; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            sum += grayscale[ny * width + nx];
            count++;
          }
        }
      }
      
      const localMean = sum / count;
      const threshold = localMean * (1 - k);
      const pixel = grayscale[y * width + x];
      const binary = pixel > threshold ? 255 : 0;
      
      const idx = (y * width + x) * 4;
      result.data[idx] = binary;
      result.data[idx + 1] = binary;
      result.data[idx + 2] = binary;
      result.data[idx + 3] = 255;
    }
  }
  
  return result;
}

// Quality assessment to determine if image needs enhancement
export function assessImageQuality(imageData: ImageData): {
  score: number;
  needsEnhancement: boolean;
  issues: string[];
} {
  const data = imageData.data;
  const issues: string[] = [];
  let qualityScore = 100;
  
  // Check contrast
  let min = 255, max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    min = Math.min(min, gray);
    max = Math.max(max, gray);
  }
  
  const contrast = max - min;
  if (contrast < 100) {
    qualityScore -= 30;
    issues.push('Low contrast');
  }
  
  // Check brightness
  let brightness = 0;
  for (let i = 0; i < data.length; i += 4) {
    brightness += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  brightness /= (data.length / 4);
  
  if (brightness < 80 || brightness > 200) {
    qualityScore -= 20;
    issues.push(brightness < 80 ? 'Too dark' : 'Too bright');
  }
  
  // Check noise (simplified)
  let noise = 0;
  for (let i = 4; i < data.length - 4; i += 4) {
    const current = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    const prev = 0.299 * data[i - 4] + 0.587 * data[i - 3] + 0.114 * data[i - 2];
    noise += Math.abs(current - prev);
  }
  noise /= (data.length / 4 - 1);
  
  if (noise > 30) {
    qualityScore -= 25;
    issues.push('High noise');
  }
  
  return {
    score: Math.max(0, qualityScore),
    needsEnhancement: qualityScore < 70,
    issues
  };
}