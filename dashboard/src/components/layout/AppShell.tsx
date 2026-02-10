import type { ReactNode } from "react";
import { TopBar } from "./TopBar";
import { BottomNav } from "./BottomNav";

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <TopBar />
      <main className="flex-1 overflow-y-auto px-4 pb-20 pt-4">
        {children}
      </main>
      <BottomNav />
    </div>
  );
}
