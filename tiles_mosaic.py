from pathlib import Path
import rasterio
from rasterio.merge import merge

tile_dir = Path("data/dem_data")
out_path = Path("data/dem.tif")

tiles = sorted(tile_dir.glob("*.tif"))
print(f"merging {len(tiles)} tiles")

srcs = [rasterio.open(p) for p in tiles]
mosaic, transform = merge(srcs)          # overlapping edges are handled for you

meta = srcs[0].meta.copy()
meta.update(
    driver="GTiff",
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    compress="lzw",
    tiled=True,
)

with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(mosaic)

for s in srcs:
    s.close()
