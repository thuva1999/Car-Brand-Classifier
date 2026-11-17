"""Phone camera -> laptop: a small web page served from the laptop on the local network.

The model runs here, on the laptop. The phone only takes the photo and shows
the answer, so no internet and no cloud service is involved. Connect the phone
and laptop to the same network (the phone's own hotspot works without
internet), then open the address printed below in the phone's browser.

    python serve.py              # port 8000
    python serve.py --port 8080

Every photo and its annotated result are saved under outputs/live/. Under each
answer the page lets you confirm or correct the brand and colour of every car;
that goes to outputs/live/feedback.csv, which gives a running live accuracy.
"""

from __future__ import annotations

import argparse
import base64
import csv
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
FEEDBACK_CSV = OUTPUT_DIR / "feedback.csv"
FEEDBACK_FIELDS = ["photo", "car_in_photo", "box", "predicted_brand", "brand_confidence",
                   "true_brand", "predicted_colour", "colour_confidence", "true_colour"]
COLOURS = ["white", "black", "grey", "red", "blue", "green", "brown", "yellow"]
NOT_SCORED = ("unknown", "other")  # can't tell / not one of the model's classes
recent: dict[str, list[dict]] = {}  # stamp -> predictions, until the phone sends feedback
feedback_lock = threading.Lock()

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
  .fb { margin-top:14px; border-top:1px solid var(--border); padding-top:12px; }
  .fbrow { display:flex; gap:8px; align-items:center; margin:8px 0; }
  .fbrow span { width:20px; font-weight:700; }
  select { flex:1; min-width:0; padding:10px 6px; font-size:16px; border-radius:8px;
           border:1px solid var(--border); background:var(--card); color:var(--text); }
  select.changed { border-color:#d1242f; border-width:2px; }
  button { width:100%; padding:14px; margin-top:8px; border:0; border-radius:10px; background:#1a7f37;
           color:#fff; font-size:18px; font-weight:600; }
  button:disabled { opacity:.6; }
  #tally { margin-top:16px; text-align:center; }
</style></head>
<body><main>
  <h1>Car brand &amp; colour</h1>
  <label class="take">Take photo<input id="camera" type="file" accept="image/*" capture="environment"></label>
  <label class="alt">or choose from gallery<input id="gallery" type="file" accept="image/*"></label>
  <div id="out"></div>
  <div id="tally" class="meta"></div>
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
        ${feedbackForm(data)}
      </div>`;
    wireFeedback(data);
  } catch (error) {
    out.innerHTML = `<div class="card error">Failed: ${esc(error.message)}</div>`;
  }
}
function options(list, chosen) {
  return list.map(v => `<option${v === chosen ? ' selected' : ''}>${esc(v)}</option>`).join('');
}
function feedbackForm(data) {
  const rows = data.cars.map((c, i) => `<div class="fbrow"><span>${i + 1}</span>
      <select data-kind="brand" data-car="${i}" data-predicted="${esc(c.brand)}">${options(data.brands, c.brand)}</select>
      <select data-kind="colour" data-car="${i}" data-predicted="${esc(c.colour)}">${options(data.colours, c.colour)}</select>
    </div>`).join('');
  return `<div class="fb"><div class="meta">Is it right? Leave correct answers as they are, change the wrong ones
      to the true brand / colour ("unknown" = can't tell, "other" = brand not in the list), then save.</div>
    ${rows}<button id="save">All correct: save</button><div id="saved" class="meta"></div></div>`;
}
function showTally(t) {
  if (!t || !t.cars) { document.getElementById('tally').textContent = ''; return; }
  const pct = (a, b) => b ? ` (${Math.round(100 * a / b)}%)` : '';
  document.getElementById('tally').textContent =
    `Checked so far: brand ${t.brand_right}/${t.brand_total}${pct(t.brand_right, t.brand_total)}` +
    ` · colour ${t.colour_right}/${t.colour_total}${pct(t.colour_right, t.colour_total)}`;
}
function wireFeedback(data) {
  const button = document.getElementById('save');
  const selects = [...out.querySelectorAll('select')];
  const refresh = () => {
    let wrong = 0;
    for (const s of selects) { const c = s.value !== s.dataset.predicted; s.classList.toggle('changed', c); wrong += c; }
    button.textContent = wrong ? `Save (${wrong} corrected)` : 'All correct: save';
    button.disabled = false;
  };
  selects.forEach(s => s.addEventListener('change', refresh));
  button.addEventListener('click', async () => {
    button.disabled = true;
    const cars = data.cars.map(() => ({}));
    for (const s of selects) cars[s.dataset.car]['true_' + s.dataset.kind] = s.value;
    try {
      const response = await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'},
                                                 body: JSON.stringify({stamp: data.stamp, cars})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || response.statusText);
      document.getElementById('saved').textContent = 'Saved. You can still change an answer and save again.';
      button.textContent = 'Saved \u2713';
      showTally(result.tally);
    } catch (error) {
      document.getElementById('saved').innerHTML = `<span class="error">Not saved: ${esc(error.message)}</span>`;
      button.disabled = false;
    }
  });
}
fetch('/tally').then(r => r.json()).then(showTally).catch(() => {});
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

    recent[stamp] = predictions
    while len(recent) > 50:
        recent.pop(next(iter(recent)))

    preview = annotated.copy()
    preview.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80)
    return jsonify(
        stamp=stamp,
        brands=sorted(pipeline.brand_names) + list(NOT_SCORED),
        colours=COLOURS + ["unknown"],
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


def read_feedback() -> list[dict]:
    if not FEEDBACK_CSV.exists():
        return []
    with open(FEEDBACK_CSV, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def tally(rows: list[dict]) -> dict:
    """Right / scored counts; "unknown" and "other" answers are not scored."""

    def count(kind: str) -> tuple[int, int]:
        scored = [row for row in rows if row[f"true_{kind}"] not in NOT_SCORED]
        return sum(row[f"true_{kind}"] == row[f"predicted_{kind}"] for row in scored), len(scored)

    brand_right, brand_total = count("brand")
    colour_right, colour_total = count("colour")
    return dict(cars=len(rows), brand_right=brand_right, brand_total=brand_total,
                colour_right=colour_right, colour_total=colour_total)


@app.get("/tally")
def get_tally():
    return jsonify(tally(read_feedback()))


@app.post("/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    stamp = str(data.get("stamp", ""))
    predictions = recent.get(stamp)
    answers = data.get("cars")
    if predictions is None:
        return jsonify(error="That photo is no longer in memory (server restarted?)."), 400
    if not isinstance(answers, list) or len(answers) != len(predictions):
        return jsonify(error="Feedback does not match the cars in that photo."), 400
    brands = set(pipeline.brand_names) | set(NOT_SCORED)
    colours = set(COLOURS) | {"unknown"}
    new_rows = []
    for number, (item, given) in enumerate(zip(predictions, answers), start=1):
        given = given if isinstance(given, dict) else {}
        true_brand, true_colour = str(given.get("true_brand")), str(given.get("true_colour"))
        if true_brand not in brands or true_colour not in colours:
            return jsonify(error=f"Unknown brand or colour for car {number}."), 400
        new_rows.append({
            "photo": f"{stamp}.jpg", "car_in_photo": number,
            "box": " ".join(str(int(value)) for value in item["box"]),
            "predicted_brand": item["brand"], "brand_confidence": f"{item['brand_confidence']:.2f}",
            "true_brand": true_brand,
            "predicted_colour": item["colour"], "colour_confidence": f"{item['colour_confidence']:.2f}",
            "true_colour": true_colour,
        })
    with feedback_lock:
        # Saving the same photo again replaces its earlier rows.
        rows = [row for row in read_feedback() if row["photo"] != f"{stamp}.jpg"] + new_rows
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEEDBACK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    wrong = [f"car {row['car_in_photo']}: {kind} {row[f'predicted_{kind}']} -> {row[f'true_{kind}']}"
             for row in new_rows for kind in ("brand", "colour")
             if row[f"true_{kind}"] != row[f"predicted_{kind}"]]
    print(f"  feedback [{stamp}]: " + ("; ".join(wrong) if wrong else "all correct"))
    return jsonify(saved=len(new_rows), tally=tally(rows))


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
