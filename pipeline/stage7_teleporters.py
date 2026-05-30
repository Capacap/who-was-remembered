"""
Stage 7: teleporter / waypoint network.

A small hand-curated set of fast-travel monuments the player can jump between.
Each one is a recognizable place-and-era the desert is worth crossing for: the
Italian Renaissance, pharaonic Egypt, Tang China. There is no auto-fill. The
network is exactly these curated points; everything between them is void left
to be wandered into, which is the rest of the map.

The anchors are NOT coordinates. Each is a label plus a list of defining people,
and the monument is placed from where those people actually landed in Stage 6.
Re-bake the placement and the teleporters move with it; nobody re-types an (x, y).
The people list also documents the anchor: you can read what defines each place.

The load-bearing constraint is that the angle axis is longitude only (Stage 6
discards latitude). So an anchor is coherent ONLY when its people are co-located
in longitude. "the Italian Renaissance" works because its figures cluster on
Italy; "World War II" cannot be one monument because Hitler, Churchill and
Hirohito span the whole disc and average to dead space at the origin. Themes
that are real but spread are split by PLACE (Abbasid Baghdad and Moorish Iberia
are two anchors, not one "Islamic Golden Age"). This stage enforces that: any
anchor whose people scatter wider than SPREAD_LIMIT in longitude is a hard
error, not a quiet mis-placement.

Two filters keep a pin honest:
- A geo-less member (geo_source is None in Stage 6) has no real coordinate; its
  angle is a per-QID hash, pure noise. Such members are dropped from the centroid
  and reported. Pinning to them would drag the monument into the void.
- The monument sits on the most cross-lingually covered (highest sitelink_count)
  surviving member: a real grave, the recognizable face of the group. For the
  tight clusters these anchors are, that point is essentially the centroid.

Prominence is used here only to choose which grave the monument sits on and which
places are worth a waypoint. It never moved a book in Stage 6 and does not here.

Run (after Stage 6):
    uv run pipeline/stage7_teleporters.py
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "placement.parquet"
PLACES_PATH = ROOT / "cache" / "places.parquet"
OUT_PATH = ROOT / "cache" / "teleporters.parquet"

# Longitude spread (max - min of members' jitter-free country longitude, in
# degrees) above which an anchor is not one place. ~40 deg is roughly the width
# of Europe; beyond it the pin is averaging across separate regions.
SPREAD_LIMIT = 40.0

# --- the authored anchor set: (label, [defining people]) ---
# Grouped by PLACE first, era second, never by theme. Names are English
# Wikipedia article titles; they resolve against placement.parquet by title.
# Edit freely, but keep each list co-located in longitude or the bake will fail.
ANCHORS: list[tuple[str, list[str]]] = [
    # the Americas arc (western longitudes, their own sky). One early-modern US
    # anchor only: "nineteenth-century America" (Lincoln, Twain, Edison) was
    # dropped because it sat a short hop from revolutionary America, and
    # teleporters that close together read as noise, not meaningful travel.
    ("Mesoamerica", ["Moctezuma II", "Nezahualcoyotl (tlatoani)", "Itzcoatl", "Ahuitzotl", "Tlacaelel"]),
    ("the Inca", ["Pachacuti", "Huayna Capac", "Topa Inca Yupanqui", "Huáscar"]),
    ("revolutionary America", ["George Washington", "Thomas Jefferson", "Benjamin Franklin", "John Adams", "Alexander Hamilton"]),
    # the Norse north (narrowed to Norway; Iceland/Greenland scatter the longitude)
    ("the Viking Age", ["Harald Hardrada", "Harald Fairhair", "Erik the Red", "Olaf Tryggvason", "Haakon Sigurdsson"]),
    # western Europe. The recent-European corner (18-19c) is the most
    # compressed region of the disc (recent = small radius, Europe = a narrow
    # longitude wedge), so anchors there pile up and stop covering meaningful
    # travel distance. Thinned to a single point, the French Revolution.
    # Dropped were "Enlightenment Paris" (Voltaire et al., co-located with the
    # Revolution at the Paris longitude), "industrial Britain" (Darwin et al.)
    # and "the German Romantics" (Beethoven, Goethe, Marx); the last two sat
    # within ~90 world-units of each other and of the Russian Empire. See
    # [[project-teleporter-network]].
    ("the French Revolution", ["Maximilien Robespierre", "Georges Danton", "Jean-Paul Marat", "Louis XVI", "Napoleon"]),
    ("Moorish Iberia", ["Averroes", "Maimonides", "Ibn Hazm", "Al-Zahrawi", "Ibn Arabi"]),
    # Italy / Rome
    ("the Roman Empire", ["Augustus", "Julius Caesar", "Cicero", "Virgil", "Marcus Aurelius", "Nero"]),
    ("the Italian Renaissance", ["Leonardo da Vinci", "Michelangelo", "Raphael", "Niccolò Machiavelli", "Galileo Galilei", "Sandro Botticelli"]),
    # Greece
    ("Classical Athens", ["Plato", "Aristotle", "Socrates", "Pericles", "Sophocles", "Euripides"]),
    # Constantinople (Byzantine then Ottoman: same longitude, different rings)
    ("Byzantine Constantinople", ["Justinian I", "Belisarius", "Basil II", "Heraclius", "Constantine the Great"]),
    ("the Ottoman Empire", ["Suleiman the Magnificent", "Mehmed II", "Selim I", "Bayezid II", "Murad IV"]),
    # Russia
    ("the Russian Empire", ["Peter the Great", "Catherine the Great", "Leo Tolstoy", "Fyodor Dostoevsky", "Alexander Pushkin"]),
    # Egypt / Mesopotamia / Arabia / Persia
    ("pharaonic Egypt", ["Ramesses II", "Tutankhamun", "Hatshepsut", "Akhenaten", "Thutmose III"]),
    ("Babylon", ["Hammurabi", "Nebuchadnezzar II", "Sargon of Akkad", "Ashurbanipal", "Tiglath-Pileser III"]),
    ("the rise of Islam", ["Muhammad", "Abu Bakr", "Umar", "Ali", "Uthman"]),
    ("Abbasid Baghdad", ["Harun al-Rashid", "Al-Mansur", "Al-Ma'mun", "Al-Kindi", "Ja'far al-Sadiq"]),
    ("Achaemenid Persia", ["Cyrus the Great", "Xerxes I", "Cambyses II", "Artaxerxes I"]),
    # India
    ("Maurya & Gupta India", ["Ashoka", "Chandragupta Maurya", "Samudragupta", "Bindusara"]),
    ("Mughal India", ["Akbar", "Aurangzeb", "Shah Jahan", "Jahangir", "Babur"]),
    # China (one longitude band, separated by ring)
    ("Confucian China", ["Confucius", "Laozi", "Mencius", "Sun Tzu", "Qin Shi Huang"]),
    ("Han China", ["Emperor Wu of Han", "Emperor Gaozu of Han", "Sima Qian", "Cao Cao", "Ban Chao"]),
    ("Tang China", ["Emperor Taizong of Tang", "Wu Zetian", "Bai Juyi", "Du Fu", "Xuanzang"]),
    ("Song China", ["Su Shi", "Wang Anshi", "Zhu Xi", "Sima Guang", "Emperor Taizu of Song"]),
    # the steppe
    ("the Mongol Empire", ["Genghis Khan", "Kublai Khan", "Ögedei Khan", "Möngke Khan", "Subutai"]),
    # Japan (one longitude band, separated by ring)
    ("Heian Japan", ["Murasaki Shikibu", "Taira no Kiyomori", "Sei Shōnagon", "Minamoto no Yoritomo"]),
    ("the age of the samurai", ["Oda Nobunaga", "Toyotomi Hideyoshi", "Tokugawa Ieyasu", "Date Masamune", "Miyamoto Musashi"]),
]

def _ord(c: int) -> str:
    if 10 <= c % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(c % 10, "th")


def century(year: float) -> str:
    y = int(round(year))
    if y > 0:
        c = (y - 1) // 100 + 1
        return f"{c}{_ord(c)} century"
    c = (-y - 1) // 100 + 1
    return f"{c}{_ord(c)} century BCE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", type=Path, default=PLACEMENT_PATH)
    parser.add_argument("--places", type=Path, default=PLACES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    start = time.perf_counter()

    t = pq.read_table(
        args.placement,
        columns=["qid", "title", "x", "y", "death_year", "sitelink_count", "country_qid", "geo_source"],
    )
    qid = t.column("qid").to_pylist()
    title = t.column("title").to_pylist()
    x = t.column("x").to_numpy()
    y = t.column("y").to_numpy()
    death = t.column("death_year").to_pylist()
    sit = t.column("sitelink_count").to_numpy()
    cc = t.column("country_qid").to_pylist()
    gs = t.column("geo_source").to_pylist()
    print(f"loaded {t.num_rows:,} placed figures from {args.placement.name}")

    # title -> row, preferring the most-covered when a title repeats (a famous
    # figure should win over an obscure namesake).
    idx_by_title: dict[str, int] = {}
    for i, tt in enumerate(title):
        j = idx_by_title.get(tt)
        if j is None or sit[i] > sit[j]:
            idx_by_title[tt] = i

    # jitter-free longitude per figure: the mean longitude of its country. The
    # baked angle carries heavy jitter and is useless for coherence checks.
    places = pq.read_table(args.places, columns=["lon", "country_qid"])
    acc: dict[str, list[float]] = defaultdict(list)
    for lon, c in zip(places.column("lon").to_numpy(), places.column("country_qid").to_pylist()):
        if c is not None and lon is not None and not math.isnan(lon):
            acc[c].append(lon)
    country_lon = {c: float(np.mean(v)) for c, v in acc.items()}

    rows = []
    for label, names in ANCHORS:
        found = [idx_by_title[n] for n in names if n in idx_by_title]
        missing = [n for n in names if n not in idx_by_title]
        geoless = [title[i] for i in found if gs[i] is None]
        members = [i for i in found if gs[i] is not None]

        if len(members) < 2:
            raise ValueError(
                f"anchor {label!r}: only {len(members)} geo-anchored members "
                f"(missing={missing} geoless={geoless}); cannot place."
            )

        lons = [country_lon[cc[i]] for i in members if cc[i] in country_lon]
        spread = (max(lons) - min(lons)) if lons else 0.0
        if spread >= SPREAD_LIMIT:
            raise ValueError(
                f"anchor {label!r}: longitude spread {spread:.0f} deg exceeds "
                f"{SPREAD_LIMIT:.0f}; its people are not co-located. Split by place."
            )

        # monument sits on the most-covered surviving member (a real grave).
        seat = max(members, key=lambda i: sit[i])
        median_era = float(np.median([death[i] for i in members]))

        flags = ""
        if missing:
            flags += f"  missing={missing}"
        if geoless:
            flags += f"  dropped-geoless={geoless}"
        print(
            f"  {label:28s} {len(members)}p spread={spread:3.0f} "
            f"r={math.hypot(x[seat], y[seat]):5.0f} on:{title[seat]}{flags}"
        )

        rows.append(dict(
            label=label,
            x=float(x[seat]),
            y=float(y[seat]),
            seat_qid=qid[seat],
            seat_title=title[seat],
            people=names,
            member_qids=[qid[i] for i in members],
            era=century(median_era),
            n_members=len(members),
            lon_spread=round(spread, 1),
        ))

    out = pa.table({
        "label": pa.array([r["label"] for r in rows], pa.string()),
        "x": pa.array([r["x"] for r in rows], pa.float64()),
        "y": pa.array([r["y"] for r in rows], pa.float64()),
        "seat_qid": pa.array([r["seat_qid"] for r in rows], pa.string()),
        "seat_title": pa.array([r["seat_title"] for r in rows], pa.string()),
        "people": pa.array([r["people"] for r in rows], pa.list_(pa.string())),
        "member_qids": pa.array([r["member_qids"] for r in rows], pa.list_(pa.string())),
        "era": pa.array([r["era"] for r in rows], pa.string()),
        "n_members": pa.array([r["n_members"] for r in rows], pa.int32()),
        "lon_spread": pa.array([r["lon_spread"] for r in rows], pa.float64()),
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out, compression="zstd")

    elapsed = time.perf_counter() - start
    print(f"placed {len(rows)} teleporters in {elapsed:.1f}s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
