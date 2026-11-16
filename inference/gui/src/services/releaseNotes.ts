import changelog from "../../../../CHANGELOG.md?raw";

export const CHANGELOG = changelog;

const heading = changelog.match(/^##\s*\[([^\]]+)\]/m);
export const LATEST_VERSION = heading ? heading[1] : "";