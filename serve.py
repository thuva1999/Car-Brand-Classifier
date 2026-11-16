"""Phone camera -> laptop: a small web page served from the laptop on the local network.

The model runs here, on the laptop. The phone only takes the photo and shows
the answer, so no internet and no cloud service is involved. Connect the phone
and laptop to the same network (the phone's own hotspot works without
internet), then open the address printed below in the phone's browser.

    python serve.py              # port 8000
    python serve.py --port 8080

Every photo and its annotated result are saved under outputs/live/.
"""

from __future__ import annotations

import argparse
import base64
import io
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from flask import Flask, jsonify, request  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from pipeline import CarPipeline, annotate_image, format_answer_line  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs" / "live"
MAX_SIDE = 2560  # phone photos are ~4000 px; YOLO works at 640 anyway

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
pipeline: CarPipeline | None = None
pipeline_lock = threading.Lock()  # one photo at a time

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Car classifier</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#16191d; --muted:#5d6670; --accent:#1f6feb; --border:#d8dde3; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1214; --card:#1a1e22; --text:#e8eaed; --muted:#9aa3ad; --accent:#4c8dff; --border:#2c3238; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:16px; font:16px/1.45 system-ui, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width:560px; margin:0 auto; }
  h1 { font-size:20px; margin:4px 0 16px; }
  label.take { display:block; text-align:center; padding:22px; border-radius:14px; background:var(--accent);
               color:#fff; font-size:20px; font-weight:600; cursor:pointer; }
  input[type=file] { display:none; }
  .alt { display:block; text-align:center; margin:10px 0 0; color:var(--accent); cursor:pointer; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px; margin-top:16px; }
  .status { color:var(--muted); }
  .answer { font-size:22px; font-weight:700; margin:4px 0 10px; word-wrap:break-word; }
  ol { margin:0; padding-left:22px; }
  li { margin:4px 0; }
  .meta { color:var(--muted); font-size:14px; }
  img { width:100%; border-radius:8px; margin-top:12px; }
  .error { color:#d1242f; }
</style></head>
<body><main>
  <h1>Car brand &amp; colour</h1>
  <label class="take">Take photo<input id="camera" type="file" accept="image/*" capture="environment"></label>
  <label class="alt">or choose from gallery<input id="gallery" type="file" accept="image/*"></label>
  <div id="out"></div>
</main>
<script>
const out = document.getElementById('out');
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
async function send(file) {
  if (!file) return;
  out.innerHTML = '<div class="card status">Uploading and classifying…</div>';
  const body = new FormData(); body.append('photo', file);
  const started = Date.now();
  try {
    const response = await fetch('/predict', {method:'POST', body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    const rows = data.cars.map(c =>
      `<li><b>${esc(c.brand)}</b> (${c.brand_confidence}%) — <b>${esc(c.colour)}</b> (${c.colour_confidence}%)` +
      ` <span class="meta">detected ${c.detect_confidence}%</span></li>`).join('');
    out.innerHTML = `<div class="card">
        <div class="meta">Answer sheet</div>
        <div class="answer">${esc(data.answer)}</div>
        ${rows ? '<ol>' + rows + '</ol>' : ''}
        <div class="meta">${data.cars.length} car(s) · model ${data.seconds}s · total ${((Date.now()-started)/1000).toFixed(1)}s</div>
        <img src="data:image/jpeg;base64,${data.annotated}" alt="Photo with detected cars boxed">
      </div>`;
  } catch (error) {
    out.innerHTML = `<div class="card error">Failed: ${esc(error.message)}</div>`;
  }
}
for (const id of ['camera', 'gallery']) {
  const input = document.getElementById(id);
  input.addEventListener('change', () => { send(input.files[0]); input.value = ''; });
}
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/predict")
def predict():
    upload = request.files.get("photo")
    if upload is None:
        return jsonify(error="No photo received."), 400
    try:
        image = ImageOps.exif_transpose(Image.open(upload.stream)).convert("RGB")
    except Exception as error:  # noqa: BLE001 - tell the phone what went wrong
        return jsonify(error=f"Could not read the image: {error}"), 400
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    started = time.time()
    with pipeline_lock:
        predictions = pipeline.predict_image(image)
    seconds = time.time() - started

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated = annotate_image(image, predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / f"{stamp}.jpg", quality=92)
    annotated.save(OUTPUT_DIR / f"{stamp}_pred.jpg", quality=88)

    answer = format_answer_line(predictions)
    print(f"\n[{stamp}] {answer}   ({seconds:.1f}s)")
    for number, item in enumerate(predictions, start=1):
        print(f"  Car {number}: {item['brand']} ({item['brand_confidence']:.0%})"
              f" — {item['colour']} ({item['colour_confidence']:.0%})")

    preview = annotated.copy()
    preview.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80)
    return jsonify(
        answer=answer,
        seconds=round(seconds, 1),
        annotated=base64.b64encode(buffer.getvalue()).decode("ascii"),
        cars=[
            {
                "brand": item["brand"],
                "colour": item["colour"],
                "brand_confidence": round(item["brand_confidence"] * 100),
                "colour_confidence": round(item["colour_confidence"] * 100),
                "detect_confidence": round(item["detect_confidence"] * 100),
            }
            for item in predictions
        ],
    )


def local_addresses() -> list[str]:
    """IPv4 addresses of this laptop on its local networks (no internet needed)."""

    addresses = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(a for a in addresses if not a.startswith("127."))


def main() -> None:
    global pipeline
    parser = argparse.ArgumentParser(description="Serve the car classifier to a phone on the local network.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Loading models...")
    pipeline = CarPipeline()
    print(f"Brand classes ({len(pipeline.brand_names)}): {', '.join(pipeline.brand_names)}")

    addresses = local_addresses()
    print("\nOpen one of these on the phone (same Wi-Fi or the phone's hotspot):")
    for address in addresses or ["<no network found - connect to the phone's hotspot first>"]:
        print(f"  http://{address}:{args.port}")
    print("If the phone cannot connect, allow Python through Windows Firewall (see README).")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
