import { Skeleton } from "../ui/skeleton";

export function LoadingSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-16 w-full" />
      ))}
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <Skeleton className="mb-2 h-4 w-24" />
      <Skeleton className="h-8 w-32" />
    </div>
  );
}
