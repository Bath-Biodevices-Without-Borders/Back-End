#from read_rivers import *
import os
import numpy as np
import cv2
from google_maps_image import get_map_image, get_blue_water_map_image, get_water_edge_image
from identification import guided_river_boundaries
from config import POINTS, GOOGLE_MAPS_API_KEY

def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def process_river_points():
    """
    Process each point in the config file:
    1. Check if river_images folder exists with aerial images
    2. If not, download satellite images at zoom level 18
    3. Generate terrain maps with blue river lines as guides
    4. Apply guided river boundary detection using the terrain maps
    5. Save only the final overlay images with river boundary in overlay_images directory
    """
    # Check if river_images directory exists
    river_images_dir = "river_images"
    if not os.path.exists(river_images_dir):
        print(f"'{river_images_dir}' directory not found. Creating it...")
        ensure_directory_exists(river_images_dir)
        use_existing_images = False
    else:
        print(f"'{river_images_dir}' directory found. Will use existing images if available.")
        use_existing_images = True
    
    # Create terrain_maps directory for storing the terrain maps with blue river lines
    terrain_maps_dir = "terrain_maps"
    ensure_directory_exists(terrain_maps_dir)
    
    # Create processing_images directory for storing intermediate results
    processing_dir = "processing_images"
    ensure_directory_exists(processing_dir)
    
    # Create overlay_images directory if it doesn't exist
    output_dir = "overlay_images"
    ensure_directory_exists(output_dir)
    
    print(f"Processing {len(POINTS)} river points...")
    
    # Process each point in the config
    for i, point in enumerate(POINTS):
        image_number = i+1
        latitude, longitude = point
        print(f"\nProcessing point {image_number}/{len(POINTS)}: {latitude}, {longitude}")
        
        # Check for existing aerial image
        aerial_filename = f"img{image_number}.png"
        aerial_path = os.path.join(river_images_dir, aerial_filename)
        
        # Path for the terrain map with blue river lines
        terrain_filename = f"terr{image_number}.png"
        terrain_path = os.path.join(terrain_maps_dir, terrain_filename)
        
        # Path for the water edge image
        edge_filename = f"edge{image_number}.png"
        edge_path = os.path.join(terrain_maps_dir, edge_filename)
        
        # Check for existing aerial image
        if use_existing_images and os.path.exists(aerial_path):
            print(f"Using existing aerial image: {aerial_path}")
            # Load the existing aerial image
            satellite_image = cv2.imread(aerial_path)
            
            if satellite_image is None:
                print(f"Error: Could not read aerial image at {aerial_path}")
                continue
        else:
            # Need to download new image
            print(f"Downloading new satellite image for point: {latitude}, {longitude}")
            try:
                # Get the satellite image
                image_data, metadata = get_map_image(
                    latitude=latitude, 
                    longitude=longitude, 
                    zoom_level=18,
                    api_key=GOOGLE_MAPS_API_KEY,
                    image_size=(640, 640)
                )
                
                # Convert image data to OpenCV format
                nparr = np.frombuffer(image_data, np.uint8)
                satellite_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save the satellite image to river_images directory
                cv2.imwrite(aerial_path, satellite_image)
                print(f"Saved satellite image to: {aerial_path}")
            except Exception as e:
                print(f"Error downloading satellite image: {e}")
                continue
        
        try:
            # Generate terrain map with blue river lines
            print(f"Generating terrain map with blue river lines for point {image_number}...")
            
            # Check if we already have the terrain map
            if os.path.exists(terrain_path):
                print(f"Using existing terrain map: {terrain_path}")
                terrain_image = cv2.imread(terrain_path)
                if terrain_image is None:
                    raise ValueError(f"Could not read terrain map at {terrain_path}")
            else:
                # Get the terrain map with blue water bodies
                blue_water_data, _ = get_blue_water_map_image(
                    latitude=latitude,
                    longitude=longitude,
                    zoom_level=18,
                    api_key=GOOGLE_MAPS_API_KEY,
                    image_size=(640, 640)
                )
                
                # Convert blue_water_data to OpenCV format
                nparr = np.frombuffer(blue_water_data, np.uint8)
                terrain_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save terrain map
                cv2.imwrite(terrain_path, terrain_image)
                print(f"Saved terrain map to: {terrain_path}")
                
                # Generate water edge image (green lines on black background)
                water_edge_data = get_water_edge_image(blue_water_data, (640, 640))
                
                # Save water edge image
                with open(edge_path, 'wb') as f:
                    f.write(water_edge_data)
                print(f"Saved water edge image to: {edge_path}")
            
            # Check if the edge image exists and read it
            if os.path.exists(edge_path):
                print(f"Using edge image as guide: {edge_path}")
                edge_image = cv2.imread(edge_path)
                if edge_image is None:
                    raise ValueError(f"Could not read edge image at {edge_path}")
                
                # Use edge image as guide if available - it has clearer green lines for the guide
                guide_image = edge_image
            else:
                # Fallback to using terrain map as guide
                guide_image = terrain_image
            
            # Use guided_river_boundaries with the appropriate guide image
            print("Detecting river boundaries using guide image...")
            results = guided_river_boundaries(
                satellite_image, 
                guide_image, 
                show_plots=False, 
                save_plots=False
            )
            
            # Save the final image with detailed boundaries to overlay_images
            overlay_image = cv2.cvtColor(results['final'], cv2.COLOR_RGB2BGR)
            overlay_image_path = os.path.join(output_dir, f"river_overlay_{image_number}.png")
            cv2.imwrite(overlay_image_path, overlay_image)
            print(f"Saved river boundary overlay to: {overlay_image_path}")
            
            # Save the original satellite image for comparison
            orig_image_path = os.path.join(output_dir, f"original_{image_number}.png")
            cv2.imwrite(orig_image_path, satellite_image)
            
            # If we want to save intermediate processing results, save them to processing_images directory
            if 'refined_boundaries' in results:
                cv2.imwrite(os.path.join(processing_dir, f"boundaries_{image_number}.png"), results['refined_boundaries'])
            if 'refined_mask' in results:
                cv2.imwrite(os.path.join(processing_dir, f"mask_{image_number}.png"), results['refined_mask'])
            
        except Exception as e:
            print(f"Error processing point {image_number}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nProcessing completed. All river boundary overlays saved to the 'overlay_images' directory.")

if __name__ == "__main__":
    process_river_points()




