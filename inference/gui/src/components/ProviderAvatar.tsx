import { useState } from "react";

import { Icon } from "./Icon";
import { Logo } from "./Logo";

/**
 * Provider artwork with a fallback.
 *
 * The registry advertises avatar files that don't all ship (custom.png, for
 * one). Rather than a broken-image glyph — or a stray initial like "C" — fall
 * back to a generic globe mark, which reads as "some other provider".
 */
export function ProviderAvatar({
  avatar,
  logo,
  size = 22,
}: {
  avatar?: string | null;
  /** The local provider is us, so it gets the Dizel mark instead. */
  logo?: boolean;
  size?: number;
}) {
  const [failed, setFailed] = useState(false);

  if (avatar && !failed) {
    return (
      <img
        className="model-icon"
        src={`./providers/${avatar}`}
        width={size}
        height={size}
        alt=""
        draggable={false}
        onError={() => setFailed(true)}
      />
    );
  }

  if (logo) return <Logo size={size} />;

  return (
    <span
      className="provider-fallback"
      style={{ width: size, height: size, borderRadius: Math.max(4, Math.round(size * 0.22)) }}
      aria-hidden="true"
    >
      <Icon name="globe" size={Math.round(size * 0.6)} />
    </span>
  );
}
