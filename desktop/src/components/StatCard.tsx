interface StatCardProps {
  label: string;
  value: string;
  hint: string;
}

export function StatCard({ label, value, hint }: StatCardProps) {
  return (
    <div className="stat-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <span className="stat-hint">{hint}</span>
    </div>
  );
}

