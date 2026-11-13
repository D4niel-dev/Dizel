import type { ReactNode, SVGProps } from "react";

export type IconName =
  | "newChat"
  | "compose"
  | "chat"
  | "folder"
  | "file"
  | "pin"
  | "archive"
  | "globe"
  | "plug"
  | "alert"
  | "trash"
  | "panel"
  | "panelCollapse"
  | "panelExpand"
  | "plus"
  | "chevronUp"
  | "chevronDown"
  | "chevronLeft"
  | "chevronRight"
  | "mic"
  | "arrowUp"
  | "settings"
  | "more"
  | "copy"
  | "refresh"
  | "check"
  | "download"
  | "close"
  | "minimize"
  | "maximize"
  | "restore"
  | "code"
  | "book"
  | "image"
  | "music"
  | "search"
  | "sparkles"
  | "keyboard"
  | "info"
  | "thumbUp"
  | "thumbDown";

const shapes: Record<IconName, ReactNode> = {
  newChat: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="4.5" />
      <path d="M12 8.6v6.8" />
      <path d="M8.6 12h6.8" />
    </>
  ),
  compose: (
    <>
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z" />
    </>
  ),
  chat: (
    <path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z" />
  ),
  folder: <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />,
  file: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3.5L21 19.5H3L12 3.5z" />
      <path d="M12 10v4" />
      <path d="M12 17h.01" />
    </>
  ),
  globe: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12h18" />
      <path d="M12 3c2.6 2.4 4 5.6 4 9s-1.4 6.6-4 9c-2.6-2.4-4-5.6-4-9s1.4-6.6 4-9z" />
    </>
  ),
  archive: (
    <>
      <rect x="3" y="4" width="18" height="4" rx="1" />
      <path d="M5 8v11a1 1 0 001 1h12a1 1 0 001-1V8" />
      <path d="M10 12h4" />
    </>
  ),
  pin: (
    <>
      <path d="M9 3h6v5.2l3 3V13H6v-1.8l3-3V3z" />
      <path d="M12 13v8" />
    </>
  ),
  trash: (
    <>
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M19 6l-1 14H6L5 6" />
      <path d="M10 11v5M14 11v5" />
    </>
  ),
  panel: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M15 4v16" />
    </>
  ),
  panelCollapse: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M9 4v16" />
      <path d="M16.5 9.5L14 12l2.5 2.5" />
    </>
  ),
  panelExpand: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M9 4v16" />
      <path d="M14 9.5L16.5 12 14 14.5" />
    </>
  ),
  plus: (
    <>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </>
  ),
  chevronUp: <path d="M18 15l-6-6-6 6" />,
  chevronDown: <path d="M6 9l6 6 6-6" />,
  chevronLeft: <path d="M15 18l-6-6 6-6" />,
  chevronRight: <path d="M9 18l6-6-6-6" />,
  mic: (
    <>
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <path d="M12 19v3" />
    </>
  ),
  arrowUp: (
    <>
      <path d="M12 19V5" />
      <path d="M5 12l7-7 7 7" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>
  ),
  more: (
    <>
      <circle cx="12" cy="5" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none" />
      <circle cx="12" cy="19" r="1.4" fill="currentColor" stroke="none" />
    </>
  ),
  copy: (
    <>
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </>
  ),
  refresh: (
    <>
      <path d="M21 3v6h-6" />
      <path d="M3 21v-6h6" />
      <path d="M3.5 9a9 9 0 0 1 14.9-3.4L21 9" />
      <path d="M20.5 15a9 9 0 0 1-14.9 3.4L3 15" />
    </>
  ),
  check: <path d="M20 6L9 17l-5-5" />,
  download: (
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5 5 5-5" />
      <path d="M12 15V3" />
    </>
  ),
  close: (
    <>
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </>
  ),
  minimize: <path d="M5 12h14" />,
  maximize: <rect x="5.5" y="5.5" width="13" height="13" rx="1.5" />,
  restore: (
    <>
      <rect x="8" y="3" width="13" height="13" rx="1.5" />
      <path d="M16 16v3.5A1.5 1.5 0 0 1 14.5 21H5A1.5 1.5 0 0 1 3.5 19.5V10A1.5 1.5 0 0 1 5 8.5h3" />
    </>
  ),
  code: (
    <>
      <path d="M16 18l6-6-6-6" />
      <path d="M8 6l-6 6 6 6" />
    </>
  ),
  music: (
    <>
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </>
  ),
  book: (
    <>
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </>
  ),
  image: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="2.5" />
      <circle cx="8.5" cy="8.5" r="1.6" />
      <path d="M21 15.5l-4.5-4.5L6 21" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="M21 21l-4.3-4.3" />
    </>
  ),
  sparkles: (
    <>
      <path d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5l4.6-1.9z" />
      <path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z" />
    </>
  ),
  keyboard: (
    <>
      <rect x="2" y="5" width="20" height="14" rx="2.5" />
      <path d="M6 9h.01M10 9h.01M14 9h.01M18 9h.01M6 13h.01M10 13h.01M14 13h.01M18 13h.01M7 17h10" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 8h.01" />
    </>
  ),
  thumbUp: (
    <>
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </>
  ),
  thumbDown: (
    <>
      <path d="M17 14V2" />
      <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
    </>
  ),
  plug: (
    <>
      <path d="M12 21v-4.2" />
      <path d="M9 7V2.5" />
      <path d="M15 7V2.5" />
      <path d="M17.5 7.5h-11l1.2 6.2a2.6 2.6 0 0 0 2.55 2.1h3.5a2.6 2.6 0 0 0 2.55-2.1z" />
    </>
  ),
};

/**
 * Solid silhouettes for the settings nav. The outline shapes are line art, so
 * simply filling them produces muddy blobs. These variants keep a recognisable
 * silhouette and cut details out with even-odd holes.
 */
const solidShapes: Partial<Record<IconName, ReactNode>> = {
  chat: (
    <path
      d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8z"
      fill="currentColor"
      stroke="none"
    />
  ),
  sparkles: (
    <>
      <path
        d="M12 3l1.9 4.6L18.5 9.5 13.9 11.4 12 16l-1.9-4.6L5.5 9.5l4.6-1.9z"
        fill="currentColor"
        stroke="none"
      />
      <path
        d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z"
        fill="currentColor"
        stroke="none"
      />
    </>
  ),
  image: (
    <path
      fillRule="evenodd"
      d="M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm2.5 4.2a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6ZM19 16.2 14.8 12 9 17.8V19h10v-2.8Z"
      fill="currentColor"
      stroke="none"
    />
  ),
  mic: (
    <>
      <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" fill="none" stroke="currentColor" />
      <path d="M12 19v3" fill="none" stroke="currentColor" />
    </>
  ),
  keyboard: (
    <path
      fillRule="evenodd"
      d="M4.5 5h15A2.5 2.5 0 0 1 22 7.5v9a2.5 2.5 0 0 1-2.5 2.5h-15A2.5 2.5 0 0 1 2 16.5v-9A2.5 2.5 0 0 1 4.5 5Zm1.5 3h2v2H6V8Zm4 0h2v2h-2V8Zm4 0h2v2h-2V8Zm4 0h2v2h-2V8ZM6 12h2v2H6v-2Zm4 0h2v2h-2v-2Zm4 0h2v2h-2v-2Zm4 0h2v2h-2v-2ZM7 16h10v2H7v-2Z"
      fill="currentColor"
      stroke="none"
    />
  ),
  info: (
    <path
      fillRule="evenodd"
      d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-1 5h2v2h-2V8Zm0 3h2v5h-2v-5Z"
      fill="currentColor"
      stroke="none"
    />
  ),
  plug: (
    <>
      <path
        d="M12 21v-3.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
      />
      <path
        d="M9 7V2.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
      />
      <path
        d="M15 7V2.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.4"
      />
      <path
        d="M17.6 7.5H6.4l1.22 6.2a2.9 2.9 0 0 0 2.84 2.3h3.08a2.9 2.9 0 0 0 2.84-2.3Z"
        fill="currentColor"
        stroke="none"
      />
    </>
  ),
};

interface IconProps extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  size?: number;
  /** Solid variant — same geometry, filled instead of outlined. */
  filled?: boolean;
}

export function Icon({
  name,
  size = 16,
  strokeWidth = 1.7,
  filled = false,
  ...rest
}: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={filled ? 1.1 : strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...rest}
    >
      {filled && solidShapes[name] ? solidShapes[name] : shapes[name]}
    </svg>
  );
}
