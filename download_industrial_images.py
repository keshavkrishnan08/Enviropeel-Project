"""
Download industrial bearing and machinery images from Wikimedia Commons.
Images are stored in reference_images/downloads/, added to DB with 50% catalog / 50% test split.
Catalog = our previous work (shown to clients when they upload); Test = held out for validation.
"""
import os
import json
import time
import random
import requests
from urllib.parse import urlencode
from catalog_metadata import lookup_reference_profile
from database import db, ImageRecord, init_db, BEARING_TYPES, COATING_STATUSES
from similarity import ImageSimilarityMatcher
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///image_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

REFERENCE_FOLDER = 'reference_images'
DOWNLOAD_SUBFOLDER = os.path.join(REFERENCE_FOLDER, 'downloads')
HEADERS = {"User-Agent": "FulcrumIndustrial/1.0 (Educational Project; Python requests)"}

WIKIMEDIA_SOURCES = [
    ("category", "Category:Roller_bearings", "roller_bearing"),
    ("category", "Category:Ball_bearings", "ball_bearing"),
    ("category", "Category:Flanges", "flange"),
    ("category", "Category:Industrial_pipes", "pipe"),
    ("category", "Category:Industrial_machinery", "machinery"),
    ("search", "pillow block bearing", "pillow_block"),
    ("search", "ball bearing industrial", "bearing"),
    ("search", "industrial pipe flange", "flange"),
    ("search", "conveyor bearing", "conveyor"),
]

CURATED_URLS = [
    ("https://upload.wikimedia.org/wikipedia/commons/6/6e/Bearing_01.JPG", "bearing_housing", "Industrial bearing housing close-up"),
    ("https://upload.wikimedia.org/wikipedia/commons/9/9d/Bearing_02.JPG", "bearing_housing", "Industrial bearing housing assembly"),
    ("https://upload.wikimedia.org/wikipedia/commons/7/72/Bearings_of_gas_turbines.JPG", "pillow_block", "Gas turbine bearings"),
    ("https://upload.wikimedia.org/wikipedia/commons/0/01/Ceramicbearing.jpg", "bearing_housing", "Ceramic bearing assembly"),
    ("https://upload.wikimedia.org/wikipedia/commons/7/75/Cannelures_2.jpg", "bearing_housing", "Bearing housing with grooves"),
    ("https://upload.wikimedia.org/wikipedia/commons/e/e6/Bearing_render.png", "bearing_housing", "Rendered bearing assembly"),
]


def fetch_wikimedia_category(category_name, limit=8):
    """Fetch image URLs from a Wikimedia Commons category."""
    urls = []
    try:
        params = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": category_name,
            "gcmlimit": limit,
            "gcmtype": "file",
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": 800,
            "format": "json",
        }
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if p.get("ns") != 6:
                continue
            info = (p.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            mime = info.get("mime", "")
            if url and "image" in mime and "svg" not in mime.lower():
                urls.append((url, p.get("title", "File:unknown.jpg").replace("File:", "")))
    except Exception as e:
        print(f"  Category fetch error: {e}")
    return urls


def fetch_wikimedia_search(query, limit=6):
    """Search Wikimedia Commons for images."""
    urls = []
    try:
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 6,
            "srlimit": limit,
            "format": "json",
        }
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=search_params,
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        titles = [s["title"] for s in data.get("query", {}).get("search", [])]
        if not titles:
            return urls
        info_params = {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 800,
            "format": "json",
        }
        r2 = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params=info_params,
            headers=HEADERS,
            timeout=15,
        )
        r2.raise_for_status()
        data2 = r2.json()
        pages = data2.get("query", {}).get("pages", {})
        for pid, p in pages.items():
            if p.get("ns") != 6:
                continue
            info = (p.get("imageinfo") or [{}])[0]
            url = info.get("url")
            mime = info.get("mime", "")
            if url and "image" in mime and "svg" not in mime.lower():
                thumb = info.get("thumburl") or url
                urls.append((thumb, p.get("title", "File:unknown.jpg").replace("File:", "")))
    except Exception as e:
        print(f"  Search fetch error: {e}")
    return urls


def download_image(url, dest_path):
    """Download an image to the target path."""
    try:
        time.sleep(1.5)
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        ext = ".jpg"
        if ".png" in url.lower():
            ext = ".png"
        if dest_path.endswith((".jpg", ".jpeg", ".png", ".gif")):
            pass
        else:
            dest_path = dest_path + ext
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return dest_path
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def assign_split(index, total):
    """Assign catalog or test split. 50% each, alternating for balance."""
    if total <= 1:
        return "catalog"
    return "catalog" if (index % 2 == 0) else "test"


def infer_bearing_type(keyword):
    kw = keyword.lower()
    if "pillow" in kw or "plummer" in kw or "ball" in kw or "roller" in kw or "bearing" in kw or "conveyor" in kw or "machinery" in kw:
        return "pillow_block"
    if "flange" in kw or "pipe" in kw:
        return "flange"
    return "pillow_block"


def main():
    os.makedirs(DOWNLOAD_SUBFOLDER, exist_ok=True)
    print("Downloading industrial bearing images from Wikimedia Commons")
    print("=" * 60)

    all_images = []

    for url, bearing_type, desc in CURATED_URLS:
        profile = lookup_reference_profile(url.split("/")[-1].split("?")[0])
        all_images.append({
            "url": url,
            "filename": url.split("/")[-1].split("?")[0],
            "bearing_type": (profile or {}).get("bearing_type") or bearing_type,
            "description": (profile or {}).get("description") or desc,
            "size_info": (profile or {}).get("size_info"),
        })

    seen_filenames = {img["filename"] for img in all_images}
    for source_type, query, bt in WIKIMEDIA_SOURCES:
        print(f"\nFetching: {query}...")
        if source_type == "category":
            pairs = fetch_wikimedia_category(query, limit=4)
        else:
            pairs = fetch_wikimedia_search(query, limit=3)
        for url, fname in pairs:
            if fname in seen_filenames:
                continue
            seen_filenames.add(fname)
            bt_clean = infer_bearing_type(bt)
            profile = lookup_reference_profile(fname)
            all_images.append({
                "url": url,
                "filename": fname,
                "bearing_type": (profile or {}).get("bearing_type") or bt_clean,
                "description": (profile or {}).get("description") or "Industrial " + bt.replace("_", " ") + " from Wikimedia Commons",
                "size_info": (profile or {}).get("size_info"),
            })
        time.sleep(0.5)

    print(f"\nTotal images to process: {len(all_images)}")
    print("Loading CLIP for embeddings...")

    with app.app_context():
        init_db()
        matcher = ImageSimilarityMatcher()

        added = 0
        failed = 0

        for i, img in enumerate(all_images):
            url = img["url"]
            safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in img["filename"])[:80]
            dest_path = os.path.join(DOWNLOAD_SUBFOLDER, safe_name)
            if not dest_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                dest_path += ".jpg"

            print(f"\n[{i+1}/{len(all_images)}] {safe_name[:50]}...")

            if not os.path.exists(dest_path):
                path = download_image(url, dest_path)
                if not path:
                    failed += 1
                    continue
                dest_path = path
                time.sleep(0.3)

            url_filename = os.path.join("downloads", os.path.basename(dest_path))

            existing = ImageRecord.query.filter(ImageRecord.filepath == dest_path).first()
            if existing:
                if not existing.split:
                    existing.split = assign_split(i, len(all_images))
                existing.bearing_type = img.get("bearing_type")
                existing.description = img.get("description")
                existing.size_info = img.get("size_info")
                db.session.commit()
                print("  Updated split & metadata")
                continue

            try:
                emb = matcher.get_image_embedding(dest_path)
                split = assign_split(i, len(all_images))
                rec = ImageRecord(
                    filename=url_filename,
                    filepath=dest_path,
                    description=img.get("description", "Industrial component"),
                    embedding=json.dumps(emb.tolist()),
                    bearing_type=img.get("bearing_type"),
                    coating_status="coated",
                    size_info=img.get("size_info"),
                    split=split,
                )
                db.session.add(rec)
                db.session.commit()
                print(f"  Added (split={split}) id={rec.id}")
                added += 1
            except Exception as e:
                db.session.rollback()
                print(f"  Error: {e}")
                failed += 1

        catalog_count = ImageRecord.query.filter(ImageRecord.split == "catalog").count()
        test_count = ImageRecord.query.filter(ImageRecord.split == "test").count()
        print("\n" + "=" * 60)
        print(f"Done. added={added} failed={failed}")
        print(f"Catalog (for matching): {catalog_count}")
        print(f"Test (held out): {test_count}")


if __name__ == "__main__":
    main()
