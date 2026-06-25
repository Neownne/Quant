#!/usr/bin/env python
"""批量 Surya OCR hotpoint 图片，增量写缓存。"""
import os, sys, json, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_hotpoint import ocr_image_surya, ocr_image, save_ocr_cache, load_ocr_cache

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'arsenal', 'hotpoint', 'ocr_cache.json')
PNG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hotpoint')

cache = load_ocr_cache()
pngs = sorted([f for f in os.listdir(PNG_DIR) if f.endswith('.png')])
print(f"Total: {len(pngs)} images", flush=True)

for i, fname in enumerate(pngs):
    dt = fname.replace('.png', '')
    if dt.startswith('206'):
        dt = '2' + dt[1:]
        print(f"  (fixed typo: {fname} → {dt})", flush=True)

    if dt in cache:
        e = cache[dt]
        if isinstance(e, dict) and e.get('engine') == 'surya' and len(e.get('records', [])) > 0:
            print(f"[{i+1}/{len(pngs)}] {dt} ✓ ({len(e['records'])} recs)", flush=True)
            continue

    filepath = os.path.join(PNG_DIR, fname)
    t0 = time.time()

    try:
        records = ocr_image_surya(filepath)
        for r in records:
            r['date'] = dt
        cache[dt] = {"engine": "surya", "records": records}
        save_ocr_cache(cache)
        print(f"[{i+1}/{len(pngs)}] {dt} → {len(records)} recs ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[{i+1}/{len(pngs)}] {dt} SURYA FAILED: {e}", flush=True)
        try:
            text = ocr_image(filepath)
            cache[dt] = {"engine": "tesseract", "text": text}
            save_ocr_cache(cache)
            print(f"  → Tesseract: {len(text)} chars", flush=True)
        except Exception as e2:
            print(f"  → Both failed: {e2}", flush=True)

print(f"\nDone! {len(cache)} entries", flush=True)
