import { useAgentStatus } from "@/api/hooks";
import { useAgent } from "@/contexts/AgentContext";
import { RegimeBadge } from "../shared/RegimeBadge";

export function TopBar() {
  const { data: agent } = useAgentStatus();
  const { agent: activeAgent, setAgent } = useAgent();

  return (
    <header className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#0a0a0f]/80 backdrop-blur-xl">
      <div className="flex h-14 items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold bg-gradient-to-r from-gold to-yellow-300 bg-clip-text text-transparent">
            MIDAS
          </span>
          <div className={`h-2 w-2 rounded-full ${agent?.running ? "bg-green-500 pulse-glow" : "bg-gray-600"}`} />
          {/* Agent Toggle Pill */}
          <div className="flex items-center bg-white/[0.04] rounded-full border border-white/[0.08] p-0.5">
            <button
              onClick={() => setAgent("llm")}
              className={`px-2 py-0.5 rounded-full text-[10px] font-medium transition-all ${
                activeAgent === "llm"
                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                  : "text-gray-500 hover:text-gray-400"
              }`}
            >
              🧠 LLM
            </button>
            <button
              onClick={() => setAgent("nollm")}
              className={`px-2 py-0.5 rounded-full text-[10px] font-medium transition-all ${
                activeAgent === "nollm"
                  ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                  : "text-gray-500 hover:text-gray-400"
              }`}
            >
              📊 NoLLM
            </button>
          </div>
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
          <span className="text-[10px] text-gray-600">v8.1</span>
        </div>
      </div>
    </header>
  );
}
