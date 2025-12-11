import json

# Path to your GeoJSON file
geojson_file = r'C:\Users\carlos\Desktop\qupathenv\results\cellpose_save\000_img.png.geojson'

# Load the GeoJSON file
with open(geojson_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Check if 'features' exist
if 'features' not in data:
    raise ValueError("The GeoJSON file does not contain 'features'")

# Extract all names from each feature
names = []
for feature in data['features']:
    # Make sure 'properties' and 'name' exist
    if 'properties' in feature and 'name' in feature['properties']:
        names.append(feature['properties']['name'])

# Sort names alphabetically
names_sorted = sorted(names)

# Print sorted names
for i, name in enumerate(names_sorted):
    print(i, name)
