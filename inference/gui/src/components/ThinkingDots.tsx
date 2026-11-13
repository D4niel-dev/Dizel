import { useEffect, useRef } from "react";
import lottie, { type AnimationItem } from "lottie-web/build/player/lottie_light";

import dotsData from "../assets/thinking-dots.json";

/**
 * Gemini's own three-dot indicator, played from their Lottie data.
 *
 * The animation is 693 frames at 60fps — 11.5 seconds — with a parent layer that
 * rotates in 120° steps and three dots carrying ~20 position keyframes each. That
 * is well past what CSS keyframes can express, which is why this ships the data
 * and a renderer rather than hand-written animation.
 *
 * The fills in the JSON are black; the colour is applied in CSS so it follows the
 * theme (`fill` as a presentation attribute loses to a CSS rule).
 */
export function ThinkingDots() {
  const host = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!host.current) return undefined;

    const animation: AnimationItem = lottie.loadAnimation({
      container: host.current,
      renderer: "svg",
      loop: true,
      autoplay: true,
      animationData: dotsData,
    });

    return () => animation.destroy();
  }, []);

  return <span className="tri-dots" ref={host} aria-hidden="true" />;
}
