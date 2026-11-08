"""Pull outfit pictures out of a PowerPoint board, in slide order, ready for prep_images.

    python3 -m mindcart.pptx_to_stimuli ~/Downloads/Grace_Fashion_Board.pptx --out stimuli/raw

- Reads the .pptx directly (it's a zip), with no extra installs.
- Keeps images in slide order (slide 1 first, top-to-bottom as they appear in the slide XML).
- Skips tiny images (icons/logos) and exact duplicates (the same picture reused).
- Writes stimuli/raw/outfit_001.jpg ... and stimuli/raw/source.csv (which slide each came from).
Then:  python3 -m mindcart.prep_images stimuli/raw stimuli/items.csv --out stimuli/norm
"""
import argparse
import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def slide_order(z):
    """Slide paths in presentation order (from presentation.xml + its rels)."""
    pres = z.read("ppt/presentation.xml").decode("utf8")
    rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf8")
    rid2target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    rid2target.update({k: v for v, k in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)})
    ids = re.findall(r'<p:sldId [^>]*r:id="(rId\d+)"', pres)
    return ["ppt/" + rid2target[i].lstrip("/").replace("ppt/", "") for i in ids if i in rid2target]


def slide_images(z, slide_path):
    """Image file paths referenced by a slide, in the order they appear in its XML."""
    xml = z.read(slide_path).decode("utf8")
    rel_path = str(PurePosixPath(slide_path).parent / "_rels" / (PurePosixPath(slide_path).name + ".rels"))
    if rel_path not in z.namelist():
        return []
    rels = z.read(rel_path).decode("utf8")
    rid2t = {}
    for m in re.finditer(r"<Relationship ([^>]*)/?>", rels):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        if a.get("Type", "").endswith("/image"):
            rid2t[a["Id"]] = a["Target"]
    out = []
    for rid in re.findall(r'r:embed="(rId\d+)"', xml):
        if rid in rid2t:
            p = (PurePosixPath(slide_path).parent / rid2t[rid]).as_posix()
            parts = []
            for seg in p.split("/"):
                if seg == "..":
                    parts.pop()
                elif seg and seg != ".":
                    parts.append(seg)
            out.append("/".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx")
    ap.add_argument("--out", default="stimuli/raw")
    ap.add_argument("--min-px", type=int, default=150, help="skip images smaller than this on both sides")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(Path(a.pptx).expanduser())
    seen, rows, skipped, dups = set(), [], 0, 0
    for s_i, sp in enumerate(slide_order(z), 1):
        for ip in slide_images(z, sp):
            data = z.read(ip)
            h = hashlib.md5(data).hexdigest()
            if h in seen:
                dups += 1; continue
            seen.add(h)
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
            except Exception:
                skipped += 1; continue          # emf/wmf/svg etc.
            if im.width < a.min_px and im.height < a.min_px:
                skipped += 1; continue
            n = len(rows) + 1
            name = f"outfit_{n:03d}.jpg"
            if im.mode in ("RGBA", "LA", "P"):   # flatten transparency onto white
                im = im.convert("RGBA"); bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); im = bg
            im.convert("RGB").save(out / name, quality=92)
            rows.append({"file": name, "slide": s_i, "source": ip, "width": im.width, "height": im.height})
    with open(out / "source.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "slide", "source", "width", "height"]); w.writeheader(); w.writerows(rows)
    print(f"Extracted {len(rows)} outfit images from {len(slide_order(z))} slides -> {out}  "
          f"(skipped {skipped} small/unsupported, {dups} duplicates)")
    if len(rows) < 40:
        print("NOTE: fewer than 40 images. Aim for 60-80 for a usable analysis.")


if __name__ == "__main__":
    main()
