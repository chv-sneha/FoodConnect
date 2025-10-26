import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: number;
  email: string;
  name: string;
  allergies: string[];
  healthConditions: string[];
}

interface AppState {
  user: User | null;
  isAuthenticated: boolean;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  recentAnalyses: any[];
  
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  addAnalysis: (analysis: any) => void;
  logout: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      token: null,
      isLoading: false,
      error: null,
      recentAnalyses: [],
      
      setUser: (user) => set({ 
        user, 
        isAuthenticated: !!user 
      }),
      
      setToken: (token) => set({ token }),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
      
      addAnalysis: (analysis) => set((state) => ({
        recentAnalyses: [analysis, ...state.recentAnalyses].slice(0, 10)
      })),
      
      logout: () => set({
        user: null,
        isAuthenticated: false,
        token: null,
        recentAnalyses: []
      })
    }),
    {
      name: 'app-storage'
    }
  )
);