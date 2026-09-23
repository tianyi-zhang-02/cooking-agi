"""Bake Natural Earth country outlines into the dot grid the crew map draws.

Run once (or when the source data changes); the site itself never downloads
anything. The output is site/static/world-dots.json: an equirectangular grid of
land cells, grouped by country, with each cell stored as a three-character
base-36 index. Countries too small to claim a cell keep their nearest one, so
every country on the list can still light up.

    python3 site/tools/bake_world.py ne_110m_admin_0_countries.geojson \
                                     ne_110m_admin_0_tiny_countries.geojson

Sources: Natural Earth 110m admin-0 countries and tiny countries (public
domain), https://github.com/nvkelso/natural-earth-vector — the second file is
where places the size of Singapore live.
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

W, H = 200, 100            # grid cells
LAT0, LAT1 = 83.0, -56.0   # top and bottom edge: drop the empty poles
OUT = Path(__file__).resolve().parents[1] / "static" / "world-dots.json"
# Natural Earth carries the formal name; a map of contributors wants the everyday one
SHORT_ZH = {"CN": "中国", "KR": "韩国", "KP": "朝鲜"}


def rings(geometry):
    """Every polygon of a feature as (exterior, [holes])."""
    if geometry["type"] == "Polygon":
        parts = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        parts = geometry["coordinates"]
    else:
        return
    for part in parts:
        yield part[0], part[1:]


def main(source, tiny=None):
    data = json.loads(Path(source).read_text())
    if tiny:                                  # point features for the small states
        data["features"] += json.loads(Path(tiny).read_text())["features"]
    lons = (np.arange(W) + 0.5) / W * 360.0 - 180.0
    lats = LAT0 - (np.arange(H) + 0.5) / H * (LAT0 - LAT1)
    grid = np.stack(np.meshgrid(lons, lats), axis=-1).reshape(-1, 2)

    countries, owned = {}, {}
    for feature in data["features"]:
        props = feature["properties"]
        code = (props.get("ISO_A2_EH") or props.get("ISO_A2") or "").strip()
        if not code or code == "-99":
            continue
        geometry = feature["geometry"]
        inside = np.zeros(len(grid), dtype=bool)
        area = -1.0
        biggest = (np.array(geometry["coordinates"], dtype=float)
                   if geometry["type"] == "Point" else None)
        for exterior, holes in rings(geometry):
            hit = MplPath(exterior).contains_points(grid)
            for hole in holes:
                hit &= ~MplPath(hole).contains_points(grid)
            inside |= hit
            span = np.ptp(np.array(exterior), axis=0)
            if span[0] * span[1] > area:      # the mainland, for the centre point
                area, biggest = span[0] * span[1], np.array(exterior).mean(axis=0)
        cells = [int(c) for c in np.flatnonzero(inside)]
        for cell in cells:                    # first country to claim a cell keeps it
            owned.setdefault(cell, code)
        # a code can appear twice (the UK's polygon and Ascension's point both say
        # GB): keep the name and centre of whichever has more land, pool the cells
        seen = countries.get(code)
        if seen and len(seen["cells"]) >= len(cells):
            seen["cells"] += cells
            continue
        countries[code] = {
            "en": props.get("NAME_EN") or props.get("NAME") or code,
            "zh": SHORT_ZH.get(code) or props.get("NAME_ZH") or props.get("NAME_EN") or code,
            "at": [round(float(biggest[1]), 2), round(float(biggest[0]), 2)],
            "cells": cells + (seen["cells"] if seen else []),
        }

    # a country with no cell of its own (an island smaller than 1.8°) borrows the
    # nearest one, so small places are not silently missing from the map
    land = np.array(sorted(owned)) if owned else np.zeros(0, dtype=int)
    land_xy = np.stack([land % W, land // W], axis=-1) if len(land) else np.zeros((0, 2))
    for code, country in countries.items():
        if country["cells"]:
            continue
        lat, lon = country["at"]
        x = (lon + 180.0) / 360.0 * W
        y = (LAT0 - lat) / (LAT0 - LAT1) * H
        if not len(land):
            continue
        nearest = int(np.argmin(((land_xy - [x, y]) ** 2).sum(axis=1)))
        country["cells"] = [int(land[nearest])]

    def pack(cells):
        return "".join(np.base_repr(c, 36).rjust(3, "0") for c in sorted(set(cells)))

    out = {
        "note": "Natural Earth 110m admin-0 (public domain), baked by site/tools/bake_world.py",
        "w": W, "h": H, "lat0": LAT0, "lat1": LAT1,
        "countries": {code: {"en": c["en"], "zh": c["zh"], "at": c["at"], "cells": pack(c["cells"])}
                      for code, c in sorted(countries.items()) if c["cells"]},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} · {len(out['countries'])} countries · "
          f"{len(owned)} land cells · {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
