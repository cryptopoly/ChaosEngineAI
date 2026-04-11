interface StatCardProps {
  label: string;
  value: string;
  hint: string;
  onClick?: () => void;
}

export function StatCard({ label, value, hint, onClick }: StatCardProps) {
  return (
    <div className={`stat-card${onClick ? " stat-card--clickable" : ""}`} onClick={onClick} role={onClick ? "button" : undefined} tabIndex={onClick ? 0 : undefined}>
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <span className="stat-hint">{hint}</span>
    </div>
  );
}
