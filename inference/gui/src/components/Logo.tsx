/**
 * The Dizel GUI app mark, from public/Dizel_GUI.png.
 *
 * The PNG carries its own alpha silhouette, so no CSS radius is applied —
 * clipping it again would cut into the artwork.
 */

interface LogoProps {
  size?: number;
  className?: string;
}

export function Logo({ size = 28, className }: LogoProps) {
  return (
    <img
      className={className ? `logo-mark ${className}` : "logo-mark"}
      src="./Dizel_GUI.png"
      width={size}
      height={size}
      alt=""
      draggable={false}
    />
  );
}

/** Model variant artwork — intentionally separate from the GUI app icon. */
export function ModelIcon({ variant, size = 16 }: { variant: string; size?: number }) {
  const file = variant.toLowerCase() === "mila" ? "Mila.png" : "Dizel.png";

  return (
    <img
      className="model-icon"
      src={`./${file}`}
      width={size}
      height={size}
      alt=""
      draggable={false}
    />
  );
}
