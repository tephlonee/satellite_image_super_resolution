import argparse
import os
from datetime import datetime, timezone
from typing import Optional, Sequence

import pystac
import stac_asset.blocking
from stac_asset import Config


DEFAULT_COLLECTION_URL = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json"


def download_capella_stac(
    output_dir: str,
    max_items: int = 10,
    cutoff_yyyymmddhhmmss: str = "20251024152849",
    assets: Optional[Sequence[str]] = None,
    collection_url: str = DEFAULT_COLLECTION_URL,
) -> int:
    cutoff = datetime.strptime(cutoff_yyyymmddhhmmss, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    os.makedirs(output_dir, exist_ok=True)

    collection = pystac.Collection.from_file(collection_url)

    downloaded_count = 0
    for link in collection.get_item_links():
        if downloaded_count >= int(max_items):
            break

        item = pystac.Item.from_file(link.absolute_href)
        if item.datetime is None or item.datetime <= cutoff:
            continue

        desired = list(assets) if assets is not None else ["preview"]
        available: list[str] = []

        for k in desired:
            if k not in item.assets:
                continue
            asset = item.assets[k]
            href = str(asset.href or "").lower()
            typ = str(asset.media_type or asset.extra_fields.get("type", "") or "").lower()
            is_tif = href.endswith(".tif") or href.endswith(".tiff") or ("image/tiff" in typ) or ("geotiff" in typ)
            if is_tif:
                available.append(k)

        if not available and desired == ["preview"]:
            for k, asset in item.assets.items():
                href = str(asset.href or "").lower()
                typ = str(asset.media_type or asset.extra_fields.get("type", "") or "").lower()
                if (href.endswith("_preview.tif") or ("preview" in k.lower())) and (
                    href.endswith(".tif") or href.endswith(".tiff") or ("image/tiff" in typ) or ("geotiff" in typ)
                ):
                    available.append(k)
                    break

        if not available:
            continue

        config = Config(include=available)

        stac_asset.blocking.download_item(
            item,
            directory=output_dir,
            config=config,
        )
        downloaded_count += 1

    return downloaded_count


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-items", type=int, default=10)
    p.add_argument("--cutoff", default="20251024152849")
    p.add_argument("--collection-url", default=DEFAULT_COLLECTION_URL)
    p.add_argument("--assets", default="preview")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if str(args.assets).strip().lower() == "all":
        assets = None
    else:
        assets = [a.strip() for a in str(args.assets).split(",") if a.strip()] if args.assets else ["preview"]
    n = download_capella_stac(
        output_dir=args.output_dir,
        max_items=args.max_items,
        cutoff_yyyymmddhhmmss=str(args.cutoff),
        assets=assets,
        collection_url=str(args.collection_url),
    )
    print(f"Downloaded {n} item(s) to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
