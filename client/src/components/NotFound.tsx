import { Link } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", window.location.pathname);
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center">
        <h1 className="text-5xl font-bold mb-4 text-foreground">404</h1>
        <p className="text-lg text-muted-foreground mb-4">Oops! Page not found</p>
        <Link to="/" className="underline text-primary hover:text-primary/80">
          Return to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
