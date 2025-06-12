import { getSafetyColor, getSafetyIcon } from '@/lib/ingredient-analyzer';

interface SafetyBadgeProps {
  score: 'safe' | 'moderate' | 'risky';
  size?: 'sm' | 'md' | 'lg';
}

export function SafetyBadge({ score, size = 'md' }: SafetyBadgeProps) {
  const sizeClasses = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  const scoreText = {
    safe: 'SAFE',
    moderate: 'MODERATE',
    risky: 'HIGH RISK'
  };

  const iconClasses = getSafetyIcon(score);
  const colorClasses = getSafetyColor(score);

  return (
    <div className={`
      inline-flex items-center space-x-2 rounded-full font-bold border
      ${sizeClasses[size]} ${colorClasses}
      shadow-lg transition-all duration-300
    `}>
      <i className={`${iconClasses}`} />
      <span>{scoreText[score]}</span>
    </div>
  );
}
