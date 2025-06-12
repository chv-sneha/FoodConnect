import { useState } from 'react';
import { Link, useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { 
  Home, 
  Camera, 
  History, 
  User, 
  Menu,
  Leaf
} from 'lucide-react';

export function TopNavigation() {
  return (
    <nav className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link href="/" className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-primary to-secondary rounded-xl flex items-center justify-center">
              <Leaf className="text-white" size={20} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">FoodSense AI</h1>
              <p className="text-xs text-gray-500">Know What You Eat</p>
            </div>
          </Link>
          <Button variant="ghost" size="icon" className="rounded-lg bg-gray-100 hover:bg-gray-200">
            <User className="text-gray-600" size={20} />
          </Button>
        </div>
      </div>
    </nav>
  );
}

export function BottomNavigation() {
  const [location] = useLocation();
  
  const navItems = [
    { href: '/', icon: Home, label: 'Home' },
    { href: '/scan', icon: Camera, label: 'Scan' },
    { href: '/history', icon: History, label: 'History' },
    { href: '/profile', icon: User, label: 'Profile' },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-2 md:hidden z-50">
      <div className="flex justify-around items-center">
        {navItems.map(({ href, icon: Icon, label }) => (
          <Link key={href} href={href}>
            <button 
              className={`flex flex-col items-center space-y-1 py-2 px-3 rounded-lg transition-colors ${
                location === href 
                  ? 'text-primary border-b-2 border-primary' 
                  : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              <Icon size={20} />
              <span className="text-xs font-medium">{label}</span>
            </button>
          </Link>
        ))}
      </div>
    </nav>
  );
}
