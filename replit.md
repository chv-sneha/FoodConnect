# FoodSense AI - AI-Powered Food Safety Platform

## Project Overview
FoodSense AI is an intelligent food safety platform that analyzes ingredient labels through image upload to provide personalized health insights and safety ratings. The platform helps users understand what they're consuming through both generic and customized analysis modes.

## Key Features
### Core Scanning Features
- **Real-time Camera Scanning**: Instant scanning like PhonePe/GPay with quality enhancement for all camera types
- **Advanced OCR**: Enhanced text extraction with image preprocessing (contrast, brightness, noise reduction)
- **Toxicity Scoring Engine**: 0-100 scoring based on chemical load, sugar content, salt content, and preservative count
- **Generic Analysis**: Open to all users - ingredient breakdown, toxicity analysis, safety ratings
- **Customized Analysis**: Personalized allergen detection and health condition warnings for registered users

### Advanced Features
- **Ingredient Substitute Suggestions**: Healthier alternatives (jaggery vs sugar, coconut oil vs palm oil, etc.)
- **Multi-language Support**: Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada with native translations
- **Voice-based Summary**: Audio playback for non-literate users in regional languages
- **Community Report Flagging**: Users can flag suspicious/dangerous products for community awareness
- **Image Quality Enhancement**: Automatic enhancement for low-quality cameras using advanced algorithms
- **Accessibility Focus**: Designed for rural users, low-end devices, and diverse literacy levels

### Planned Features (Roadmap)
- **Barcode Scanner Integration**: Auto-fetch product data to avoid OCR errors
- **FSSAI & FDA Database Sync**: Real-time updates on product certifications and banned substances
- **AI Chatbot Assistant**: Answer questions like "Is this good for diabetic patients?"
- **Offline Mode**: Basic scanning and alerts for rural areas with limited internet

## Architecture
- **Frontend**: React with TypeScript, Tailwind CSS, shadcn/ui components
- **Backend**: Express.js with TypeScript
- **Storage**: In-memory storage (MemStorage) for development
- **OCR**: Tesseract.js for text extraction from images
- **Authentication**: Simple username/password system

## User Flow
1. **Landing Page**: Shows both Generic and Customized Analysis options
2. **Generic Analysis**: Available without login - basic ingredient analysis
3. **Customized Analysis**: Requires login/registration with health profile setup
4. **Registration**: Includes allergen and health condition selection during signup
5. **Analysis Results**: Personalized warnings based on user profile

## Color Palette & Theme
- **Primary**: Green (#22c55e) - represents safety and health
- **Secondary**: Blue (#3b82f6) - represents trust and technology
- **Safety Colors**: Green (Safe), Orange (Moderate), Red (Risky)
- **Design**: Modern, clean interface with gradient backgrounds and card-based layouts

## User Preferences
- Maintain existing color scheme and visual design
- Keep tagline "Know What You Eat"
- Implement two-tier analysis system (Generic/Customized)
- Require authentication for personalized features
- Include health profile setup during registration

## Recent Changes (August 2025)
- ✅ **Real-time Camera Scanning** - Implemented PhonePe/GPay-style instant camera scanning
- ✅ **Image Quality Enhancement** - Advanced OCR preprocessing for low-quality cameras (Gaussian blur, CLAHE, adaptive binarization)
- ✅ **Toxicity Scoring Engine** - Comprehensive 0-100 scoring based on chemical load, sugar, salt, and additives
- ✅ **Ingredient Substitute Suggestions** - Healthier alternatives (jaggery vs sugar, coconut oil vs palm oil)
- ✅ **Multi-language Support** - Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada
- ✅ **Voice-based Summary** - Audio playback for non-literate users in regional languages
- ✅ **Community Report Flagging** - Users can flag suspicious products for community awareness
- ✅ **Enhanced Analysis Results** - Detailed breakdown with progress bars, risk factors, recommendations
- ✅ **Advanced Authentication** - Comprehensive health profile setup during registration
- ✅ **Accessibility Features** - Works with any camera quality, enhanced for rural users

## Technical Stack
- React + TypeScript + Vite
- Express.js backend
- Tailwind CSS + shadcn/ui
- Tesseract.js for OCR
- Wouter for routing
- TanStack Query for state management