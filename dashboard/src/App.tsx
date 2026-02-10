import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import Dashboard from "./pages/Dashboard";
import Portfolio from "./pages/Portfolio";
import Performance from "./pages/Performance";
import Signals from "./pages/Signals";
import Market from "./pages/Market";
import Journal from "./pages/Journal";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/signals" element={<Signals />} />
        <Route path="/market" element={<Market />} />
        <Route path="/journal" element={<Journal />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </AppShell>
  );
}
