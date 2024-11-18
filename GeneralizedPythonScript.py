# -*- coding: utf-8 -*-
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import Polygon, Point, mapping

# Log the start of the process
print("Starting the process for creating city and hillside grids...")

# Load the boundaries
city_boundary_path = r"C:\Path\To\City\Boundary"
hillside_boundary_path = r"C:\Path\To\Hillside\Boundary"

city_boundary = gpd.read_file(city_boundary_path)
hillside_boundary = gpd.read_file(hillside_boundary_path)

# Ensure CRS match
if city_boundary.crs != hillside_boundary.crs:
    hillside_boundary = hillside_boundary.to_crs(city_boundary.crs)

# Load the elevation data
elevation_raster_path = r"C:\Path\To\Raster\File"
with rasterio.open(elevation_raster_path) as src:
    raster_crs = src.crs  # Get CRS of raster

    # Reproject boundaries if needed
    if city_boundary.crs != raster_crs:
        city_boundary = city_boundary.to_crs(raster_crs)
        hillside_boundary = hillside_boundary.to_crs(raster_crs)

    # Clip elevation data to city boundary for grid creation
    out_image, out_transform = rasterio.mask.mask(src, [mapping(city_boundary.geometry[0])], crop=True)
    elevation_data = out_image[0]  # Single-band elevation data
    rows, cols = np.where(elevation_data != -999999)

    # Get adjacent and diagonal neighbors' elevations
    def get_neighbors(row, col, elevation_data):
        adjacent = []
        diagonal = []
        directions_adjacent = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        directions_diagonal = [(-1, -1), (1, 1), (-1, 1), (1, -1)]
        
        # Collect adjacent neighbors(order N, S, W, E)
        for dr, dc in directions_adjacent:
            nr, nc = row + dr, col + dc
            if 0 <= nr < elevation_data.shape[0] and 0 <= nc < elevation_data.shape[1]:
                adjacent.append(elevation_data[nr, nc])
            else:
                adjacent.append(-999999.0)
        
        # Collect diagonal neighbors(order NW, SE, NE, SW)
        for dr, dc in directions_diagonal:
            nr, nc = row + dr, col + dc
            if 0 <= nr < elevation_data.shape[0] and 0 <= nc < elevation_data.shape[1]:
                diagonal.append(elevation_data[nr, nc])
            else:
                diagonal.append(-999999.0)

        return adjacent, diagonal
    
    def calculate_gradients(center_elevation, adjacent_elevations, diagonal_elevations):
        gradients = []
        valid_gradients = []
        
        #Check for invalid neighbor values
        if len(diagonal_elevations) != 4 or len(adjacent_elevations) != 4:
            print(f"Diagonal Elevations: {diagonal_elevations}")
            print(f"Adjacent Elevations: {adjacent_elevations}")
        
        #Order neighbors in a clockwise fashion starting from NW
        ordered_directions = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W']
        combined_elevations = [
            diagonal_elevations[0],  # NW
            adjacent_elevations[0],  # N
            diagonal_elevations[2],  # NE
            adjacent_elevations[3],  # E
            diagonal_elevations[1],  # SE
            adjacent_elevations[1],  # S
            diagonal_elevations[3],  # SW
            adjacent_elevations[2],  # W
        ]

    
        for elev, direction in zip(combined_elevations, ordered_directions):
            if elev > -999999.0:
                # Distance depends on whether it's adjacent or diagonal
                distance = 30 if direction in ['N', 'E', 'S', 'W'] else 42.4264
                gradient = (center_elevation - elev) / distance
                gradients.append(gradient)
                valid_gradients.append(gradient)
            else:
                gradients.append(-999999.0)
    
        # Compute statistics only for valid gradients
        if valid_gradients:
            absolute_gradients = [abs(gradient) if gradient > -999999 else gradient for gradient in gradients]
            avg_gradient = np.mean(valid_gradients)
            max_gradient = max(valid_gradients)
            min_gradient = min(valid_gradients)
            abs_max_gradient = abs(max_gradient)
            abs_min_gradient = abs(min_gradient)
            stddev_gradient = np.std(valid_gradients)
        else:
            # Handle case where no valid gradients exist
            absolute_gradients = []
            avg_gradient = None
            max_gradient = None
            min_gradient = None
            abs_max_gradient = None
            abs_min_gradient = None
            stddev_gradient = None
    
        return {
            'gradients': gradients,  # Includes -999999 values to preserve order
            'absolute_gradients': absolute_gradients,
            'avg_gradient': avg_gradient,
            'max_gradient': max_gradient,
            'min_gradient': min_gradient,
            'abs_max_gradient': abs_max_gradient,
            'abs_min_gradient': abs_min_gradient,
            'stddev_gradient': stddev_gradient,
        }
    
    grid_cells = []

    # Process each valid grid cell
    for idx, (row, col) in enumerate(zip(rows, cols)):
        # Define grid cell bounds
        top_left = src.xy(row, col)
        bottom_right = src.xy(row + 1, col + 1)
        polygon = Polygon([
            (top_left[0], top_left[1]),
            (bottom_right[0], top_left[1]),
            (bottom_right[0], bottom_right[1]),
            (top_left[0], bottom_right[1]),
            (top_left[0], top_left[1])
        ])
        
        # Calculate the center point of the grid cell
        center_x = (top_left[0] + bottom_right[0]) / 2
        center_y = (top_left[1] + bottom_right[1]) / 2
        center_point = Point(center_x, center_y)
        
        # Check if the center point is within the hillside area
        in_hillside = hillside_boundary.contains(center_point).any()

        # Get elevation value for this grid cell
        cell_elevation = max(elevation_data[row, col], 0)

        # Find adjacent and diagonal neighbors' elevation values
        adj_elevations, diag_elevations = get_neighbors(row, col, elevation_data)

        # Convert neighbor lists to strings for CSV compatibility
        adj_elevations_str = ",".join(map(str, adj_elevations))
        diag_elevations_str = ",".join(map(str, diag_elevations))
        
        gradient_data = calculate_gradients(cell_elevation, adj_elevations, diag_elevations)
        
        gradients_str = ",".join(map(str, gradient_data['gradients']))
        abs_gradients_str = ",".join(map(str, gradient_data['absolute_gradients']))
        
        valid_adj_elevations = [elev for elev in adj_elevations if elev != -999999]
        valid_diag_elevations = [elev for elev in diag_elevations if elev != -999999]
        valid_elevations = [cell_elevation] + valid_adj_elevations + valid_diag_elevations
        
        min_elevation = min(valid_elevations) if valid_elevations else None
        max_elevation = max(valid_elevations) if valid_elevations else None
        abs_min_elevation = abs(min_elevation) if min_elevation is not None else None
        abs_max_elevation = abs(max_elevation) if max_elevation is not None else None

        # Store all required info
        grid_cells.append({
            'geometry': polygon,
            'in_hillside': 1 if in_hillside else 0,
            'elevation': cell_elevation,
            'adjacent_elevations': adj_elevations_str,
            'diagonal_elevations': diag_elevations_str,
            'min_elevation': min_elevation,
            'max_elevation': max_elevation,
            'abs_min_elevation': abs_min_elevation,
            'abs_max_elevation': abs_max_elevation,
            'gradients': gradients_str,
            'absolute_gradients': abs_gradients_str,
            'max_gradient': gradient_data['max_gradient'],
            'min_gradient': gradient_data['min_gradient'],
            'avg_gradient': gradient_data['avg_gradient'],
            'stddev_gradient': gradient_data['stddev_gradient'],
            'abs_max_gradient': gradient_data['abs_max_gradient'],
            'abs_min_gradient': gradient_data['abs_min_gradient'],
        })

        # Log progress every 1000 cells processed
        if idx % 10000 == 0 and idx > 0:
            print(f"Processed {idx} grid cells...")

# Convert list of grid cells to a GeoDataFrame
grid_gdf = gpd.GeoDataFrame(grid_cells, crs=city_boundary.crs)

# Export grid data to a CSV file
output_csv_path = r"C:\Output\Path\For\csv.csv"
columns_to_export = [
    'in_hillside', 'elevation', 'adjacent_elevations', 'diagonal_elevations',
    'min_elevation', 'max_elevation', 'abs_min_elevation', 'abs_max_elevation',
    'gradients', 'absolute_gradients', 'max_gradient', 'min_gradient', 
    'avg_gradient', 'stddev_gradient', 'abs_max_gradient', 
    'abs_min_gradient', 'geometry'
]
grid_gdf[columns_to_export].to_csv(output_csv_path, index=False)
print(f"Expanded grid data saved to {output_csv_path}")

# Also save as shapefile if required for spatial analysis
output_shapefile_path = r"C:\Output\Path\For\Shapefile.shp"
grid_gdf.to_file(output_shapefile_path)
print(f"Shapefile saved to {output_shapefile_path}")




