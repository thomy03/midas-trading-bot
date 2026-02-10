import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Wallet,
  TrendingUp,
  Zap,
  FlaskConical,
} from "lucide-react";
import { cn } from "@/lib/utils";

const tabs = [
  { to: "/", icon: LayoutDashboard, label: "Home" },
  { to: "/portfolio", icon: Wallet, label: "Portfolio" },
  { to: "/strategies", icon: FlaskConical, label: "A/B Test" },
  { to: "/signals", icon: Zap, label: "Signals" },
  { to: "/performance", icon: TrendingUp, label: "Perf" },
];

export function BottomNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-border bg-[#0a0a0f]/95 backdrop-blur-sm safe-area-pb">
      <div className="flex h-14 items-center justify-around">
        {tabs.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex flex-col items-center gap-0.5 px-3 py-1 text-[10px]",
                isActive ? "text-gold" : "text-gray-500"
              )
            }
          >
            <Icon size={20} />
            <span>{label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
