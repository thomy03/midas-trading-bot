import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
} from "recharts";

interface Slice {
  name: string;
  value: number;
}

interface Props {
  data: Slice[];
  colors?: string[];
}

const DEFAULT_COLORS = [
  "#d4a017",
  "#3b82f6",
  "#22c55e",
  "#8b5cf6",
  "#f59e0b",
  "#ef4444",
  "#06b6d4",
  "#ec4899",
  "#14b8a6",
  "#f97316",
];

export function DonutChart({ data, colors = DEFAULT_COLORS }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-gray-500">
        No data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={80}
          paddingAngle={2}
          dataKey="value"
        >
          {data.map((_, i) => (
            <Cell key={i} fill={colors[i % colors.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: "#1a1a24",
            border: "1px solid #2a2a3a",
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value: number) => [`${value.toFixed(1)}%`, "Allocation"]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
