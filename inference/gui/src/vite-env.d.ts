/// <reference types="vite/client" />

import type { AttachmentFile, SidecarInfo, SystemStats, UpdateState } from "./types";

declare global {
  interface Window {
    dizel?: {
      platform: string;
      getSidecar(): Promise<SidecarInfo | null>;
      onSidecarReady(cb: (info: SidecarInfo) => void): () => void;
      onSidecarError(cb: (message: string) => void): () => void;
      attention(): void;
      onSidecarDown(cb: (info: { code: number | null; signal: string | null }) => void): () => void;
      restartSidecar(): Promise<{
        ok: boolean;
        info?: SidecarInfo;
        error?: string;
      }>;
      window: {
        minimize(): void;
        toggleMaximize(): void;
        close(): void;
        state(): Promise<{ maximized: boolean; fullscreen: boolean }>;
        onMaximizeChange(cb: (maximized: boolean) => void): () => void;
      };
      openExternal(url: string): Promise<boolean>;
      updates?: {
        getState(): Promise<UpdateState>;
        check(): Promise<UpdateState>;
        download(): Promise<UpdateState>;
        install(): Promise<{ ok: boolean; opened?: boolean }>;
        openRelease(): Promise<boolean>;
        setAutoCheck(enabled: boolean): Promise<UpdateState>;
        onStatus(cb: (state: UpdateState) => void): () => void;
      };
      readClipboardText(): Promise<string>;
      pickFiles(kind: "attachment" | "checkpoint" | "image"): Promise<string[]>;
      pickFolder(): Promise<string[]>;
      saveJson(
        defaultName: string,
        json: string,
      ): Promise<string | { error: string } | null>;
      onNewChat(callback: () => void): void;
      pathForFile(file: File): string;
      readDataUrl(
        path: string,
      ): Promise<{ ok: boolean; dataUrl?: string; size?: number; error?: string }>;
      revealInFolder(path: string): Promise<{ ok: boolean; error?: string }>;
      openPath(path: string): Promise<{ ok: boolean; error?: string }>;
      saveText(options: {
        defaultName: string;
        contents: string;
      }): Promise<{ ok: boolean; path?: string; canceled?: boolean; error?: string }>;
      readTextFile(path: string): Promise<AttachmentFile>;
      systemStats(): Promise<SystemStats>;
      setZoom(factor: number): void;
    };
  }
}

export {};
