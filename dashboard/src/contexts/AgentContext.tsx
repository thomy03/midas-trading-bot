import React, { createContext, useContext, useState } from "react";
type Agent = "llm" | "nollm";
const AgentContext = createContext<{ agent: Agent; setAgent: (a: Agent) => void }>({ agent: "llm", setAgent: () => {} });
export const useAgent = () => useContext(AgentContext);
export function AgentProvider({ children }: { children: React.ReactNode }) {
  const [agent, setAgent] = useState<Agent>(() => (localStorage.getItem("midas-agent") as Agent) || "llm");
  const set = (a: Agent) => { setAgent(a); localStorage.setItem("midas-agent", a); };
  return <AgentContext.Provider value={{ agent, setAgent: set }}>{children}</AgentContext.Provider>;
}
