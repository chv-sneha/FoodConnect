import Tesseract from 'tesseract.js';

export interface OCRResult {
  text: string;
  confidence: number;
}

export async function extractTextFromImage(imageFile: File): Promise<OCRResult> {
  try {
    const { data } = await Tesseract.recognize(imageFile, 'eng', {
      logger: m => console.log(m)
    });
    
    return {
      text: data.text,
      confidence: data.confidence
    };
  } catch (error) {
    console.error('OCR Error:', error);
    throw new Error('Failed to extract text from image');
  }
}

export function preprocessImageForOCR(file: File): Promise<File> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Apply preprocessing filters
      if (ctx) {
        ctx.filter = 'contrast(150%) brightness(110%)';
        ctx.drawImage(img, 0, 0);
        
        canvas.toBlob((blob) => {
          if (blob) {
            const processedFile = new File([blob], file.name, { type: file.type });
            resolve(processedFile);
          } else {
            resolve(file);
          }
        }, file.type);
      } else {
        resolve(file);
      }
    };
    
    img.src = URL.createObjectURL(file);
  });
}
