import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn, formatPct, formatCurrency } from "@/lib/utils";

interface PnlBadgeProps {
  value: number;
  mode?: "pct" | "currency";
  size?: "sm" | "md" | "lg";
}

export function PnlBadge({ value, mode = "pct", size = "md" }: PnlBadgeProps) {
  const positive = value > 0;
  const zero = value === 0;

  const sizeClasses = {
    sm: "text-xs",
    md: "text-sm",
    lg: "text-lg font-semibold",
  };

  const Icon = zero ? Minus : positive ? TrendingUp : TrendingDown;
  const display = mode === "pct" ? formatPct(value) : formatCurrency(value);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1",
        sizeClasses[size],
        positive ? "text-green-400" : zero ? "text-gray-400" : "text-red-400"
      )}
    >
      <Icon size={size === "lg" ? 18 : 14} />
      {display}
    </span>
  );
}
