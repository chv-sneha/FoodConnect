import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "@/context/AuthContext";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { OfflineIndicator } from "@/hooks/useOffline";
import { LoadingOverlay } from "@/components/ui/loading";
import { useStore } from "@/store/useStore";
import Home from "@/pages/home";
import Auth from "@/pages/auth";
import Scan from "@/pages/scan";
import Generic from "@/pages/generic";
import Results from "@/pages/results";
import Profile from "@/pages/profile";
import ProfileEdit from "@/pages/profile-edit";
import NotFound from "@/pages/not-found";
import SmartGroceryList from "@/components/SmartGroceryList";
import SmartRecipeList from "@/components/SmartRecipeList";
import AiMealBudgetPlanner from "@/components/AiMealBudgetPlanner";
import MealPrepPlanner from "@/components/MealPrepPlanner";
import HealingRecipes from "@/components/HealingRecipes";
import AiHealthForecast from "@/components/AiHealthForecast";
import ConsumerRights from "@/components/ConsumerRights";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/auth" component={Auth} />
      <Route path="/generic" component={Generic} />
      <Route path="/scan">
        <ProtectedRoute><Scan /></ProtectedRoute>
      </Route>
      <Route path="/customized">
        <ProtectedRoute><Scan /></ProtectedRoute>
      </Route>
      <Route path="/results/:id" component={Results} />
      <Route path="/history">
        <ProtectedRoute><Profile /></ProtectedRoute>
      </Route>
      <Route path="/profile">
        <ProtectedRoute><Profile /></ProtectedRoute>
      </Route>
      <Route path="/profile/edit">
        <ProtectedRoute><ProfileEdit /></ProtectedRoute>
      </Route>
      <Route path="/smart-grocery-list" component={SmartGroceryList} />
      <Route path="/smart-recipe-list" component={SmartRecipeList} />
      <Route path="/ai-meal-budget-planner" component={AiMealBudgetPlanner} />
      <Route path="/meal-prep-planner" component={MealPrepPlanner} />
      <Route path="/healing-recipes" component={HealingRecipes} />
      <Route path="/ai-health-forecast" component={AiHealthForecast} />
      <Route path="/consumer-rights" component={ConsumerRights} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  const { isLoading } = useStore();
  
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <TooltipProvider>
            <LoadingOverlay isLoading={isLoading} text="Loading...">
              <OfflineIndicator />
              <Toaster />
              <Router />
            </LoadingOverlay>
          </TooltipProvider>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
