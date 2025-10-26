import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { Link } from "wouter";

export const BackButton = () => {
  return (
    <div className="mb-6">
      <Link href="/">
        <Button variant="ghost" className="flex items-center space-x-2 text-white hover:text-white/80">
          <ArrowLeft size={16} />
          <span>Back to Home</span>
        </Button>
      </Link>
    </div>
  );
};