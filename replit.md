# FoodSense AI - AI-Powered Food Safety Platform

## Project Overview
FoodSense AI is an intelligent food safety platform that analyzes ingredient labels through image upload to provide personalized health insights and safety ratings. The platform helps users understand what they're consuming through both generic and customized analysis modes.

## Key Features
- **Generic Analysis**: Open to all users - provides ingredient breakdown, toxicity analysis, and safety ratings
- **Customized Analysis**: Requires user login - personalized allergen detection and health condition warnings
- **OCR Integration**: Extracts text from food product images using Tesseract.js
- **Safety Scoring**: Color-coded system (Green/Orange/Red) for easy understanding
- **FSSAI Verification**: Checks Indian food safety compliance

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
- Updated homepage to show Generic vs Customized Analysis options
- Implemented authentication system with login/register flows
- Added health profile setup during registration
- Restructured navigation based on user authentication state
- Maintained existing color palette and design elements

## Technical Stack
- React + TypeScript + Vite
- Express.js backend
- Tailwind CSS + shadcn/ui
- Tesseract.js for OCR
- Wouter for routing
- TanStack Query for state management