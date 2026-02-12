import { NavLink } from "react-router-dom";
import { LayoutDashboard, Wallet, FlaskConical, Zap, TrendingUp } from "lucide-react";
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
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-white/[0.06] bg-[#0a0a0f]/90 backdrop-blur-xl safe-area-pb">
      <div className="flex h-16 items-center justify-around max-w-lg mx-auto">
        {tabs.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex flex-col items-center gap-1 px-4 py-1.5 rounded-xl transition-all duration-200",
                isActive
                  ? "text-gold bg-gold/10"
                  : "text-gray-500 hover:text-gray-300"
              )
            }
          >
            <Icon size={20} />
            <span className="text-[10px] font-medium">{label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
