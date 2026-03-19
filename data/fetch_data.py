import pystac
import stac_asset
import stac_asset.blocking
from stac_asset import Config
from datetime import datetime, timezone
import os

# =========================
# CONFIG
# =========================
CUTOFF = datetime.strptime("20251024152849", "%Y%m%d%H%M%S")
CUTOFF = CUTOFF.replace(tzinfo=timezone.utc)
ASSETS_TO_DOWNLOAD = ["thumbnail", "preview"]
OUTPUT_DIR = "../batch_downloads"
MAX_ITEMS = 10   # 👈 control how many items to download

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build a Config that filters to only the desired asset keys
download_config = Config(include=ASSETS_TO_DOWNLOAD)

# =========================
# LOAD COLLECTION
# =========================
collection = pystac.Collection.from_file(
    "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"
)

downloaded_count = 0

# =========================
# LOOP + FILTER + LIMIT
# =========================
for link in collection.get_item_links():
    if downloaded_count >= MAX_ITEMS:
        break

    item = pystac.Item.from_file(link.absolute_href)

    print(f"Found item: {item.id} | datetime: {item.datetime}")

    if item.datetime is None:
        continue

    if item.datetime <= CUTOFF:
        continue

    # Defensively check which requested assets actually exist on this item
    available = [k for k in ASSETS_TO_DOWNLOAD if k in item.assets]
    if not available:
        print(f"  Skipping {item.id} — none of {ASSETS_TO_DOWNLOAD} found in assets: {list(item.assets.keys())}")
        continue

    print(f"Downloading item {downloaded_count+1}/{MAX_ITEMS}: {item.id} (assets: {available})")

    stac_asset.blocking.download_item(
        item,
        directory=OUTPUT_DIR,
        config=Config(include=available),
    )

    downloaded_count += 1