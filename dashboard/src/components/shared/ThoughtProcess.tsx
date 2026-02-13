import { Brain, CheckCircle, XCircle, AlertTriangle, Zap, BarChart3, TrendingUp, Shield } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThoughtStep {
  icon: "brain" | "technical" | "fundamental" | "ml" | "orchestrator" | "check" | "reject" | "warning";
  label: string;
  detail: string;
  status: "pass" | "fail" | "neutral" | "boost";
  score?: number;
}

interface ThoughtProcessProps {
  steps: ThoughtStep[];
  finalDecision: "ACCEPTED" | "REJECTED" | "WATCHING";
  reasoning?: string;
}

const iconMap = {
  brain: Brain,
  technical: BarChart3,
  fundamental: TrendingUp,
  ml: Zap,
  orchestrator: Brain,
  check: CheckCircle,
  reject: XCircle,
  warning: AlertTriangle,
};

const statusColors = {
  pass: "text-green-400 border-green-500/30 bg-green-500/10",
  fail: "text-red-400 border-red-500/30 bg-red-500/10",
  neutral: "text-gray-400 border-gray-500/30 bg-gray-500/10",
  boost: "text-purple-400 border-purple-500/30 bg-purple-500/10",
};

const statusDotColors = {
  pass: "bg-green-400",
  fail: "bg-red-400",
  neutral: "bg-gray-500",
  boost: "bg-purple-400",
};

export function ThoughtProcess({ steps, finalDecision, reasoning }: ThoughtProcessProps) {
  const decisionColor = finalDecision === "ACCEPTED" 
    ? "text-green-400 border-green-500/30" 
    : finalDecision === "REJECTED"
    ? "text-red-400 border-red-500/30"
    : "text-yellow-400 border-yellow-500/30";

  return (
    <div className="relative pl-8 space-y-3">
      {/* Vertical line */}
      <div className="absolute left-[15px] top-2 bottom-2 w-[2px] bg-gradient-to-b from-purple-500/40 via-purple-500/20 to-transparent" />

      {steps.map((step, i) => {
        const Icon = iconMap[step.icon] || Brain;
        return (
          <div key={i} className="relative fade-up" style={{ animationDelay: `${i * 0.08}s` }}>
            {/* Dot on the line */}
            <div className={cn("absolute -left-8 top-1.5 h-[10px] w-[10px] rounded-full border-2 border-surface", statusDotColors[step.status])} />
            
            <div className={cn("rounded-xl border px-3 py-2", statusColors[step.status])}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon size={13} />
                  <span className="text-xs font-semibold">{step.label}</span>
                </div>
                {step.score !== undefined && (
                  <span className="text-xs font-bold">{step.score.toFixed(0)}</span>
                )}
              </div>
              <p className="mt-1 text-[11px] opacity-80">{step.detail}</p>
            </div>
          </div>
        );
      })}

      {/* Final decision */}
      <div className="relative fade-up" style={{ animationDelay: `${steps.length * 0.08}s` }}>
        <div className={cn("absolute -left-8 top-1.5 h-[10px] w-[10px] rounded-full border-2 border-surface", 
          finalDecision === "ACCEPTED" ? "bg-green-400" : finalDecision === "REJECTED" ? "bg-red-400" : "bg-yellow-400"
        )} />
        <div className={cn("rounded-xl border-2 px-3 py-2 font-semibold", decisionColor)}>
          <div className="flex items-center gap-2">
            {finalDecision === "ACCEPTED" ? <CheckCircle size={14} /> : finalDecision === "REJECTED" ? <XCircle size={14} /> : <AlertTriangle size={14} />}
            <span className="text-sm">Verdict: {finalDecision}</span>
          </div>
          {reasoning && (
            <p className="mt-1 text-[11px] font-normal opacity-80">{reasoning}</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** Build thought steps from signal data */
export function buildThoughtSteps(signal: any): { steps: any[]; finalDecision: any; reasoning: string } {
  const steps: ThoughtStep[] = [];
  
  // Step 1: Technical pillar
  const tech = signal.pillar_technical ?? signal.technical_score ?? 0;
  steps.push({
    icon: "technical",
    label: "Technical Analysis (55%)",
    detail: `Score: ${tech.toFixed(1)}/25 — ${tech > 15 ? "Strong momentum & trend alignment" : tech > 10 ? "Moderate technical signals" : "Weak technicals"}`,
    status: tech > 15 ? "pass" : tech > 10 ? "neutral" : "fail",
    score: tech,
  });

  // Step 2: Fundamental pillar
  const fund = signal.pillar_fundamental ?? signal.fundamental_score ?? 0;
  steps.push({
    icon: "fundamental",
    label: "Fundamental Analysis (45%)",
    detail: `Score: ${fund.toFixed(1)}/25 — ${fund > 15 ? "Solid fundamentals, good value" : fund > 10 ? "Average fundamentals" : "Poor fundamental profile"}`,
    status: fund > 15 ? "pass" : fund > 10 ? "neutral" : "fail",
    score: fund,
  });

  // Step 3: ML Gate
  const ml = signal.ml_score ?? signal.ml_gate_score ?? null;
  if (ml !== null && ml !== undefined) {
    steps.push({
      icon: "ml",
      label: "ML Gate (Random Forest)",
      detail: ml < 40 ? `ML Score: ${ml.toFixed(0)} — BLOCKED (< 40 threshold)` : ml > 60 ? `ML Score: ${ml.toFixed(0)} — BOOSTED (> 60)` : `ML Score: ${ml.toFixed(0)} — Pass-through`,
      status: ml < 40 ? "fail" : ml > 60 ? "boost" : "neutral",
      score: ml,
    });
  }

  // Step 4: Orchestrator
  const orch = signal.orchestrator_decision ?? signal.reasoning ?? null;
  if (orch) {
    steps.push({
      icon: "orchestrator",
      label: "LLM Orchestrator",
      detail: typeof orch === "string" ? orch.slice(0, 120) : "Orchestrator reviewed sentiment + news context",
      status: signal.accepted ? "pass" : "neutral",
    });
  }

  const totalScore = signal.confidence_score ?? signal.score_at_entry ?? (tech + fund);
  const accepted = signal.accepted ?? (signal.recommendation?.includes("BUY") || signal.recommendation?.includes("STRONG"));
  const rejected = signal.rejected ?? signal.recommendation?.includes("HOLD") ?? false;

  return {
    steps,
    finalDecision: accepted ? "ACCEPTED" : rejected ? "REJECTED" : "WATCHING",
    reasoning: signal.reasoning ?? `Combined score: ${totalScore.toFixed(1)}`,
  };
}
