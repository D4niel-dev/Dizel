// Renders an SVG (Textual's screenshot export) to PNG so it can be reviewed.
const { app, BrowserWindow } = require("electron");
const fs = require("fs");

const src = process.argv[2];
const dest = process.argv[3];

app.disableHardwareAcceleration();
// Windows display scaling otherwise crops the export: the window is sized in
// CSS pixels but capturePage works in device pixels.
app.commandLine.appendSwitch("force-device-scale-factor", "1");

app.whenReady().then(async () => {
  const svg = fs.readFileSync(src, "utf8");
  // Textual's export carries the real terminal grid in viewBox; the width and
  // height attributes belong to the inner glyph ruler and crop the composer away.
  const viewBox = /viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"/.exec(svg);
  const width = Math.ceil(Number(viewBox?.[3] ?? 1200));
  const height = Math.ceil(Number(viewBox?.[4] ?? 800));

  const win = new BrowserWindow({
    width,
    height,
    show: false,
    useContentSize: true,
    webPreferences: { offscreen: true, zoomFactor: 1 },
  });

  const html = `<!doctype html><html><body style="margin:0;background:#201d1d;overflow:hidden">${svg}</body></html>`;
  await win.loadURL("data:text/html;charset=utf-8," + encodeURIComponent(html));
  await new Promise((r) => setTimeout(r, 800));

  // Capture exactly the SVG box so there is no dead canvas around it.
  const image = await win.webContents.capturePage({ x: 0, y: 0, width, height });
  // capturePage returns the window surface, which can be larger than the SVG
  // at non-100% DPI. Crop back to the SVG box so coordinates match the grid.
  const cropped = image.crop({ x: 0, y: 0, width, height });
  fs.writeFileSync(dest, cropped.toPNG());
  console.log("rendered", width, "x", height, "->", dest);
  app.quit();
});
