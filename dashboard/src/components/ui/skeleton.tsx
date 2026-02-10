import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

export function Skeleton({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-lg bg-surface-3",
        className
      )}
      {...props}
    />
  );
}
