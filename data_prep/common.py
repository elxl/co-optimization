import geopandas as gpd

# TLC zones inside Manhattan that are not modelled: Randalls Island, Roosevelt Island and the
# Governor's/Ellis/Liberty Island zones (no road or subway connection to the island).
EXCLUDED_ZONES = [194, 202, 105, 103, 104]

GTFS_NAME = 'f-dr5r-nyctsubway-latest'  # GTFS folder (unzipped) and <GTFS_NAME>.zip inside raw_dir


def load_manhattan_zones(raw_dir):
    """Modelled Manhattan taxi zones, in the shapefile CRS (EPSG:2263)."""
    zones = gpd.read_file(f'{raw_dir}/taxi_zones/taxi_zones.shp')
    return zones[(zones['borough'] == 'Manhattan') & ~zones['LocationID'].isin(EXCLUDED_ZONES)].copy()


def zone_centroids(zones):
    """Zone centroids in WGS84 (centroid taken after reprojection, as in the original pipeline)."""
    zones = zones.to_crs('EPSG:4326')
    zones['geometry'] = zones.geometry.centroid
    return zones
