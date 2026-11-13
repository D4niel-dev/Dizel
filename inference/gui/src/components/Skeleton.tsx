interface SkeletonProps {
  width?: number | string;
  height?: number;
  radius?: number;
  className?: string;
}

/**
 * Shimmer placeholder used while the sidecar is still starting. It mirrors the
 * shape of the real content so the first paint is a preview of the UI rather
 * than an empty shell.
 */
export function Skeleton({ width = "100%", height = 12, radius = 4, className }: SkeletonProps) {
  return (
    <span
      className={className ? `skel ${className}` : "skel"}
      style={{ width, height, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}
