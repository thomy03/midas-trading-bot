import { useAgentStatus } from "@/api/hooks";
import { RegimeBadge } from "../shared/RegimeBadge";
import { Menu } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

export function TopBar() {
  const { data: agent } = useAgentStatus();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-[#0a0a0f]/90 backdrop-blur-sm">
      <div className="flex h-12 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Link to="/" className="text-lg font-bold text-gold">
            MIDAS
          </Link>
          <div
            className={`h-2 w-2 rounded-full ${
              agent?.running ? "bg-green-500 animate-pulse" : "bg-gray-600"
            }`}
          />
        </div>

        <div className="flex items-center gap-2">
          {agent && (
            <span
              className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                agent.llm_enabled !== false
                  ? "bg-purple-500/20 text-purple-300"
                  : "bg-gray-500/20 text-gray-400"
              }`}
            >
              {agent.llm_enabled !== false ? "LLM" : "NO LLM"}
            </span>
          )}
          {agent && <RegimeBadge regime={agent.market_regime} />}
          <button
            className="rounded-lg p-1.5 text-gray-400 hover:bg-surface-2 hover:text-gray-200"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <Menu size={18} />
          </button>
        </div>
      </div>

      {menuOpen && (
        <nav className="border-t border-border bg-surface px-4 py-2">
          <Link
            to="/journal"
            className="block rounded-lg px-3 py-2 text-sm text-gray-300 hover:bg-surface-2"
            onClick={() => setMenuOpen(false)}
          >
            Trade Journal
          </Link>
          <Link
            to="/settings"
            className="block rounded-lg px-3 py-2 text-sm text-gray-300 hover:bg-surface-2"
            onClick={() => setMenuOpen(false)}
          >
            Settings
          </Link>
        </nav>
      )}
    </header>
  );
}
