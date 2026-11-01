#!/bin/bash
# fetch_earth_data.sh [dir]: downloads the public-domain NASA source imagery for the `earth` scene
# (not committed; ~150 MB; needs curl, unzip, python3 + PIL + pyshp for the mask).
# Then: cargo run --release -p planet --example earth_bake -- [dir] [dir/earth.pln]   (~70 s, 364 MB pack)
#       cargo run --release -p render-ash --example earth -- --shot all --out out/quality/steps
# Sources (NASA Visible Earth / Earth Observatory, free to re-use, credit "NASA Earth Observatory"):
#   Blue Marble Next Generation, July 2004 (Stöckli et al. 2005), 21600x10800, no relief shading
#   GEBCO_08-derived topography 0..6400 m and bathymetry -8000..0 m (Jesse Allen, NASA EO; GEBCO/BODC)
#   Black Marble 2016 night lights, grayscale 3 km (NASA EO / Suomi NPP VIIRS DNB, Miguel Román, NASA GSFC)
#   Blue Marble clouds 8192x4096 (NASA GSFC, R. Stöckli; MODIS)
#   Yale Bright Star Catalogue, 5th rev. ed. (Hoffleit & Warren 1991), CDS catalogue V/50
#   NOAA-20 VIIRS Corrected Reflectance (true colour) 2026-07-15, the day of the DSCOVR EPIC
#     reference frame: NASA GIBS WMS (EOSDIS, open data, credit "NASA EOSDIS GIBS / Worldview");
#     32 tiles of 45 deg x 2048 px (~26 MB) -> tools/earth_clouds.py -> clouds_viirs.png (dated
#     cloud coverage, the KTX2 bake's default cloud source; BAKE_CLOUD_SRC=bmng for Blue Marble)
#     and VIIRS_NOAA20_Cloud_Top_Height_Day (same pass, ~50 MB with the above) -> cth_viirs.png
#   Natural Earth 1:10m land, lakes, glaciated areas, Antarctic ice shelves (public domain, naturalearthdata.com)
set -euo pipefail
T=$(dirname "$(readlink -f "$0")")
D=$(readlink -f "${1:-/workspace/data/earth}"); mkdir -p "$D"; cd "$D"
B=https://eoimages.gsfc.nasa.gov/images/imagerecords
get() { [ -s "$2" ] || curl -fL --retry 3 -o "$2.part" "$B/$1" && mv -f "$2.part" "$2" 2>/dev/null || true; [ -s "$2" ] && echo "ok $2 $(stat -c %s "$2")"; }
get 74000/74092/world.200407.3x21600x10800.jpg  bmng_july.jpg
get 73000/73934/gebco_08_rev_elev_21600x10800.png topo.png
get 73000/73963/gebco_08_rev_bath_21600x10800.png bath.png
get 144000/144897/BlackMarble_2016_3km_gray_geo.tif night.tif
get 57000/57747/cloud_combined_8192.tif clouds.tif
N=https://naciscdn.org/naturalearth/10m/physical
for z in ne_10m_land ne_10m_lakes ne_10m_glaciated_areas ne_10m_antarctic_ice_shelves_polys; do
  [ -s $z.zip ] || curl -fL --retry 3 -o $z.zip $N/$z.zip; mkdir -p ne; unzip -oq $z.zip -d ne; echo "ok $z"; done
[ -s bsc5.dat ] || { curl -fL --retry 3 -o bsc5.dat.gz https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz && gunzip -f bsc5.dat.gz; }
echo "ok bsc5.dat"
sha256sum bsc5.dat bmng_july.jpg topo.png bath.png night.tif clouds.tif ne_10m_*.zip > SHA256SUMS
[ -s mask.png ] || python3 "$T/earth_mask.py" "$D"
DATE=${VIIRS_DATE:-2026-07-15}; V=viirs_${DATE//-/}; mkdir -p $V
L=VIIRS_NOAA20_CorrectedReflectance_TrueColor
for j in 0 1 2 3; do for i in 0 1 2 3 4 5 6 7; do
  la0=$((90 - (j+1)*45)); la1=$((90 - j*45)); lo0=$((-180 + i*45)); f=$V/t_${j}_${i}.jpg
  [ -s $f ] || curl -fs --retry 4 -o $f "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=$L&CRS=EPSG:4326&BBOX=$la0,$lo0,$la1,$((lo0+45))&WIDTH=2048&HEIGHT=2048&FORMAT=image/jpeg&TIME=$DATE" &
done; wait; done
# cloud-top height, same pass (8 x 4 PNG tiles of 1024 px, ~4.9 km) + its GIBS colormap
C=VIIRS_NOAA20_Cloud_Top_Height_Day; mkdir -p $V/cth
[ -s $V/cth/colormap.xml ] || curl -fs --retry 4 -o $V/cth/colormap.xml https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_VIIRS_Cloud_Top_Height.xml
for j in 0 1 2 3; do for i in 0 1 2 3 4 5 6 7; do
  la0=$((90 - (j+1)*45)); la1=$((90 - j*45)); lo0=$((-180 + i*45)); f=$V/cth/t_${j}_${i}.png
  [ -s $f ] || curl -fs --retry 4 -o $f "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS=$C&CRS=EPSG:4326&BBOX=$la0,$lo0,$la1,$((lo0+45))&WIDTH=1024&HEIGHT=1024&FORMAT=image/png&TIME=$DATE" &
done; wait; done
echo "ok $V ($(ls $V/t_*.jpg | wc -l) + $(ls $V/cth/t_*.png | wc -l) tiles)"
[ -s clouds_viirs.png ] && [ -s cth_viirs.png ] || python3 "$T/earth_clouds.py" "$D" "${DATE//-/}"
