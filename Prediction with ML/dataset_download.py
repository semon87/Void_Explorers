import earthaccess

# Log in to EarthData
auth = earthaccess.login(persist=True)

# Define collection and parameters
short_name = "TEMPO_NO2_L3"   # dataset short name
version = "V03"

# Point of interest (example: central USA)
POI_lat = 38.0
POI_lon = -96.0
date_start = "2024-09-01 00:00:00"
date_end = "2024-09-01 23:59:59"

# Search by point of interest
POI_results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    temporal=(date_start, date_end),
    point=(POI_lon, POI_lat),
)

print(len(POI_results))   # number of matching files

# Search by bounding box (wider area)
dlat = 5.0
dlon = 6.0
bbox_results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    temporal=(date_start, date_end),
    bounding_box=(POI_lon - dlon, POI_lat - dlat,
                  POI_lon + dlon, POI_lat + dlat),
)

print(len(bbox_results))

# Inspect a sample result
print(POI_results[0])
print(POI_results[-1].data_links()[0])   # direct file URL

# Download selected granules (example: 2 files)
files = earthaccess.download(POI_results[8:10], local_path=".")
