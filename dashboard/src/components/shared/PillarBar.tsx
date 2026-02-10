import { PILLAR_COLORS } from "@/lib/constants";

interface PillarBarProps {
  technical?: number;
  fundamental?: number;
  sentiment?: number;
  news?: number;
  max?: number;
}

export function PillarBar({
  technical = 0,
  fundamental = 0,
  sentiment = 0,
  news = 0,
  max = 25,
}: PillarBarProps) {
  const pillars = [
    { key: "technical", label: "T", value: technical },
    { key: "fundamental", label: "F", value: fundamental },
    { key: "sentiment", label: "S", value: sentiment },
    { key: "news", label: "N", value: news },
  ];

  return (
    <div className="flex items-center gap-1">
      {pillars.map((p) => {
        const pct = Math.min((p.value / max) * 100, 100);
        return (
          <div key={p.key} className="flex flex-col items-center">
            <div className="h-1.5 w-8 rounded-full bg-surface-3 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${pct}%`,
                  backgroundColor:
                    PILLAR_COLORS[p.key as keyof typeof PILLAR_COLORS],
                }}
              />
            </div>
            <span className="mt-0.5 text-[8px] text-gray-500">{p.label}</span>
          </div>
        );
      })}
    </div>
  );
}
