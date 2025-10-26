import { ReactNode } from 'react';
import { useMobile } from '@/hooks/use-mobile';

interface ResponsiveContainerProps {
  children: ReactNode;
  className?: string;
}

export function ResponsiveContainer({ children, className = '' }: ResponsiveContainerProps) {
  const isMobile = useMobile();
  
  return (
    <div className={`
      ${isMobile ? 'px-4 py-2' : 'px-8 py-4'}
      ${className}
    `}>
      <div className={`
        mx-auto
        ${isMobile ? 'max-w-sm' : 'max-w-4xl'}
      `}>
        {children}
      </div>
    </div>
  );
}