import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import type { PortfolioHistory } from "@/api/types";

interface Props {
  data: PortfolioHistory;
}

export function EquityCurve({ data }: Props) {
  const daily = data.daily_history;
  const dates = Object.keys(daily).sort();

  if (dates.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-gray-500">
        No history data yet
      </div>
    );
  }

  // Normalize to percentage returns from start
  const startValue = daily[dates[0]].open;
  const startSpy = daily[dates[0]].spy_open || 1;

  const chartData = dates.map((date) => {
    const d = daily[date];
    return {
      date: date.slice(5), // MM-DD
      portfolio: ((d.close / startValue - 1) * 100),
      spy: d.spy_close ? ((d.spy_close / startSpy - 1) * 100) : 0,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData}>
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: "#6b7280" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v.toFixed(0)}%`}
          width={40}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1a1a24",
            border: "1px solid #2a2a3a",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value: number, name: string) => [
            `${value.toFixed(2)}%`,
            name === "portfolio" ? "Midas" : "SPY",
          ]}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          formatter={(value) => (value === "portfolio" ? "Midas" : "SPY")}
        />
        <Line
          type="monotone"
          dataKey="portfolio"
          stroke="#d4a017"
          strokeWidth={2}
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="spy"
          stroke="#6b7280"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="4 2"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
