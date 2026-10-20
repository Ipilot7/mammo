"""Capture screenshots of the viewer by driving headless Chrome over the DevTools protocol.

Usage (server must be running):
    python scripts/screenshots.py --url http://localhost:8000 --image a.png --image-low b.png --dicom c.dcm
"""
import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


class Page:
    def __init__(self, ws):
        self.ws, self.next_id = ws, 0

    async def send(self, method, **params):
        self.next_id += 1
        mid = self.next_id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error in {expr!r}: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return r.get("result", {}).get("value")

    async def wait(self, expr, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expr):
                return
            await asyncio.sleep(0.2)
        raise TimeoutError(expr)

    async def viewport(self, w, h, mobile=False):
        await self.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=2 if mobile else 1,
                        mobile=mobile)
        await asyncio.sleep(0.4)

    async def shot(self, path):
        await asyncio.sleep(0.4)
        data = (await self.send("Page.captureScreenshot", format="png"))["data"]
        Path(path).write_bytes(base64.b64decode(data))
        print("saved", path)

    async def upload(self, file):
        doc = await self.send("DOM.getDocument")
        node = await self.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#file")
        await self.js("window.__fn = document.getElementById('filename').textContent")
        await self.send("DOM.setFileInputFiles", files=[str(Path(file).resolve())], nodeId=node["nodeId"])
        await self.wait("document.getElementById('busy').hidden && "
                        "document.getElementById('filename').textContent !== window.__fn")

    async def hover(self, selector):
        box = await self.js(f"(() => {{ const r = document.querySelector('{selector}').getBoundingClientRect();"
                            " return [r.x + r.width / 2, r.y + r.height / 2]; })()")
        await self.send("Input.dispatchMouseEvent", type="mouseMoved", x=box[0], y=box[1])
        return box

    async def click(self, selector):
        x, y = await self.hover(selector)
        for t in ("mousePressed", "mouseReleased"):
            await self.send("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)


async def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                               f"--user-data-dir={profile}", "--window-size=1440,900", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        async with websockets.connect(ws_url, max_size=None) as ws:
            p = Page(ws)
            await p.send("Page.enable")
            await p.viewport(1440, 900)
            await p.send("Page.navigate", url=args.url)
            await p.wait("document.readyState === 'complete'")
            await p.shot(out / "01_empty.png")

            await p.upload(args.image)
            await p.shot(out / "02_result_high_risk.png")

            await p.hover("#findings li")
            await p.shot(out / "03_hover_finding.png")
            await p.click("#findings li")
            await p.shot(out / "04_zoom_to_finding.png")

            await p.js("document.getElementById('fit').click();"
                       "for (const [id, v] of [['brightness', 130], ['contrast', 170]]) {"
                       " const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event('input')); }")
            await p.click("#invert")
            await p.shot(out / "05_brightness_contrast_invert.png")
            await p.click("#invert")
            await p.click("#reset")
            await p.click("#toggle")
            await p.shot(out / "06_overlay_hidden.png")
            await p.click("#toggle")

            await p.upload(args.image_low)
            await p.shot(out / "07_result_low_risk.png")

            if args.dicom:
                await p.upload(args.dicom)
                await p.shot(out / "08_dicom.png")

            await p.viewport(390, 844, mobile=True)
            await p.upload(args.image)
            await p.shot(out / "09_mobile.png")
    finally:
        chrome.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--image", required=True, help="image with a high-risk finding")
    ap.add_argument("--image-low", required=True, help="image with a low-risk finding")
    ap.add_argument("--dicom", default=None)
    ap.add_argument("--out", default="docs/screenshots")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
