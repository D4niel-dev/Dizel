import type { CSSProperties } from "react";

export type MaskIconName =
  | "edit-3"
  | "new-chat-2"
  | "workspace-2"
  | "image"
  | "layout"
  | "search"
  | "archive"
  | "book"
  | "activity";

interface MaskIconProps {
  name: MaskIconName;
  size?: number;
  className?: string;
  style?: CSSProperties;
}

/**
 * Renders one of the desktop app's own SVG files.
 *
 * They're drawn with a hard-coded black stroke, so they're used as a CSS mask
 * over `currentColor` instead of as an <img> — that way they inherit the theme
 * colour and hover states for free.
 */
export function MaskIcon({ name, size = 20, className, style }: MaskIconProps) {
  const url = `url("./icons/${name}.svg")`;

  return (
    <span
      className={className ? `mask-icon ${className}` : "mask-icon"}
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        WebkitMaskImage: url,
        maskImage: url,
        ...style,
      }}
    />
  );
}
