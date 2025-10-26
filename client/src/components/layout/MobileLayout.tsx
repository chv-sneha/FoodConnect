import { ReactNode } from 'react';
import { useMobile } from '@/hooks/use-mobile';

interface MobileLayoutProps {
  children: ReactNode;
}

export function MobileLayout({ children }: MobileLayoutProps) {
  const isMobile = useMobile();
  
  return (
    <div className={`min-h-screen ${isMobile ? 'px-4 py-2' : 'px-8 py-4'}`}>
      <div className={`mx-auto ${isMobile ? 'max-w-sm' : 'max-w-4xl'}`}>
        {children}
      </div>
    </div>
  );
}