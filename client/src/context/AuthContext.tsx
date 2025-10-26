import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut, onAuthStateChanged } from 'firebase/auth';
import { doc, setDoc, getDoc } from 'firebase/firestore';
import { auth, db } from '../lib/firebase';

interface User {
  uid: string;
  email: string;
  name: string;
  allergies: string[];
  healthConditions: string[];
  age?: number;
  activityLevel?: string;
  healthGoal?: string;
  dislikedIngredients?: string[];
  cuisineType?: string;
  dietaryPreferences?: string;
  productCategories?: string;
  emergencyContact?: string;
}

interface AuthContextType {
  user: User | null;
  login: (email: string, password: string) => Promise<boolean>;
  register: (userData: {
    email: string;
    password: string;
    name: string;
    allergies: string[];
    healthConditions: string[];
    age?: number;
    activityLevel?: string;
    healthGoal?: string;
    dislikedIngredients?: string[];
    cuisineType?: string;
    dietaryPreferences?: string;
    productCategories?: string;
    emergencyContact?: string;
  }) => Promise<boolean>;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const userDoc = await getDoc(doc(db, 'users', firebaseUser.uid));
        if (userDoc.exists()) {
          const userData = { uid: firebaseUser.uid, ...userDoc.data() } as User;
          setUser(userData);
          setIsAuthenticated(true);
        }
      } else {
        setUser(null);
        setIsAuthenticated(false);
      }
    });
    return unsubscribe;
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      return true;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const register = async (userData: {
    email: string;
    password: string;
    name: string;
    allergies: string[];
    healthConditions: string[];
    age?: number;
    activityLevel?: string;
    healthGoal?: string;
    dislikedIngredients?: string[];
    cuisineType?: string;
    dietaryPreferences?: string;
    productCategories?: string;
    emergencyContact?: string;
  }): Promise<boolean> => {
    try {
      const { password, ...userProfile } = userData;
      const userCredential = await createUserWithEmailAndPassword(auth, userData.email, password);
      await setDoc(doc(db, 'users', userCredential.user.uid), userProfile);
      return true;
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    }
  };

  const logout = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <AuthContext.Provider value={{
      user,
      login,
      register,
      logout,
      isAuthenticated,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}