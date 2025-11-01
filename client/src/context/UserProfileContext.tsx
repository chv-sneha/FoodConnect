import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useAuth } from './AuthContext';

interface UserProfile {
  dislikedIngredients?: string[];
  dietaryPreferences?: string;
  healthGoals?: string[];
}

interface UserProfileContextType {
  userProfile: UserProfile | null;
  updateProfile: (profile: Partial<UserProfile>) => void;
  isLoading: boolean;
}

const UserProfileContext = createContext<UserProfileContextType | undefined>(undefined);

export function UserProfileProvider({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (user) {
      setUserProfile({
        dislikedIngredients: user.dislikedIngredients || [],
        dietaryPreferences: user.dietaryPreferences || '',
        healthGoals: user.healthGoal ? [user.healthGoal] : []
      });
    } else {
      setUserProfile(null);
    }
  }, [user]);

  const updateProfile = (profile: Partial<UserProfile>) => {
    setUserProfile(prev => prev ? { ...prev, ...profile } : profile as UserProfile);
  };

  return (
    <UserProfileContext.Provider value={{ userProfile, updateProfile, isLoading }}>
      {children}
    </UserProfileContext.Provider>
  );
}

export function useUserProfile() {
  const context = useContext(UserProfileContext);
  if (context === undefined) {
    throw new Error('useUserProfile must be used within a UserProfileProvider');
  }
  return context;
}