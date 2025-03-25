import requests
import os
from typing import Tuple, Optional, Dict, Literal
from config import GOOGLE_MAPS_API_KEY
import cv2
import numpy as np
from datetime import datetime

def calculate_image_metadata(
    latitude: float,
    longitude: float,
    zoom_level: int,
    image_size: Tuple[int, int]) -> Dict[str, float]:
 
    # Earth's radius in meters
    R_MAJOR = 6378137.0
    R_MINOR = 6356752.3142
    RATIO = R_MINOR / R_MAJOR
    
    # Calculate meters per pixel at the equator
    meters_per_pixel = 156543.03392 * np.cos(np.radians(latitude)) / (2 ** zoom_level)
    
    # Adjust for latitude (pixels get smaller as you move away from equator)
    meters_per_pixel *= np.sqrt(1 - (RATIO * np.sin(np.radians(latitude))) ** 2)
    
    # Calculate total area covered by the image
    width_meters = meters_per_pixel * image_size[0]
    height_meters = meters_per_pixel * image_size[1]
    area_square_meters = width_meters * height_meters
    
    return {
        'meters_per_pixel': meters_per_pixel,
        'width_meters': width_meters,
        'height_meters': height_meters,
        'area_square_meters': area_square_meters
    }

def get_map_image(
    latitude: float,  # lat of centre 
    longitude: float,  # long of centre
    zoom_level: int = 21,  # default to maximum zoom
    api_key: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    maptype: str = 'satellite',
    style_strings: Optional[list] = None
) -> Tuple[bytes, Dict[str, float]]:
    """
    Get a Google Maps static image centered on a coordinate with a specified zoom level.
    
    Args:
        latitude (float): Latitude of the center point
        longitude (float): Longitude of the center point
        zoom_level (int): Zoom level (0-21, default 21 for maximum detail)
        api_key (str, optional): Google Maps API key
        image_size (tuple): Size of the output image in pixels (width, height)
        maptype (str): Type of map to return ('satellite', 'roadmap', 'terrain', 'hybrid')
        style_strings (list, optional): List of style strings to customize map appearance
    
    Returns:
        tuple: (image_data, metadata)
    """
    # Get API key from config or environment if not provided
    if api_key is None:
        api_key = GOOGLE_MAPS_API_KEY or os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("Google Maps API key not provided and not found in config or environment variables")
    
    # Ensure zoom level is within valid range
    zoom_level = max(0, min(21, zoom_level))
    
    # Calculate image metadata
    metadata = calculate_image_metadata(latitude, longitude, zoom_level, image_size)
    
    # Construct the URL for the Static Maps API
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    # Request a taller image to account for the attribution
    request_height = image_size[1] + 40
    
    # Parameters for the API request
    params = {
        'center': f"{latitude},{longitude}",
        'zoom': zoom_level,
        'size': f"{image_size[0]}x{request_height}",
        'maptype': maptype,
        'key': api_key,
        'scale': 5  # Request 2x resolution for better quality
    }
    
    # Add style parameter if provided
    if style_strings:
        # Google Maps Static API expects multiple style parameters with the same name
        for style_string in style_strings:
            # Add to existing list or create new one
            if 'style' in params:
                params['style'].append(style_string)
            else:
                params['style'] = [style_string]
    
    # Make the request
    response = requests.get(base_url, params=params)
    
    # Print URL for debugging (without API key)
    debug_url = response.url.replace(api_key, "API_KEY_HIDDEN")
    print(f"Debug URL: {debug_url}")
    
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Crop the Google attribution from the bottom and resize to desired dimensions
    return crop_and_resize_image(response.content, image_size), metadata

def get_blue_water_map_image(
    latitude: float,
    longitude: float,
    zoom_level: int = 21,
    api_key: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640)
) -> Tuple[bytes, Dict[str, float]]:
    """
    Get a Google Maps static image with water bodies represented in blue color,
    but with all labels and place names removed.
    
    Args:
        latitude (float): Latitude of the center point
        longitude (float): Longitude of the center point
        zoom_level (int): Zoom level (0-21, default 21 for maximum detail)
        api_key (str, optional): Google Maps API key
        image_size (tuple): Size of the output image in pixels (width, height)
    
    Returns:
        tuple: (image_data, metadata)
    """
    # Get API key from config or environment if not provided
    if api_key is None:
        api_key = GOOGLE_MAPS_API_KEY or os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("Google Maps API key not provided and not found in config or environment variables")
    
    # Ensure zoom level is within valid range
    zoom_level = max(0, min(21, zoom_level))
    
    # Calculate image metadata
    metadata = calculate_image_metadata(latitude, longitude, zoom_level, image_size)
    
    # Request a taller image to account for the attribution
    request_height = image_size[1] + 40
    
    # Construct the URL for the Static Maps API with style parameter directly in URL
    # The style parameter for "no labels" needs to be properly URL encoded
    base_url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom_level}&size={image_size[0]}x{request_height}&maptype=terrain&style=feature:all|element:labels|visibility:off&key={api_key}&scale=5"
    
    # Make the request
    response = requests.get(base_url)
    
    # Print URL for debugging (without API key)
    debug_url = base_url.replace(api_key, "API_KEY_HIDDEN")
    print(f"Debug URL: {debug_url}")
    
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Crop the Google attribution from the bottom and resize to desired dimensions
    return crop_and_resize_image(response.content, image_size), metadata

def get_water_edge_image(blue_water_image_data: bytes, image_size: Tuple[int, int]) -> bytes:
    """
    Create an image that highlights only the edges of water in green on a black background.
    Uses the specific RGBA value (144,218,238,255) to detect water.
    
    Args:
        blue_water_image_data (bytes): The image data containing the blue water map
        image_size (tuple): Size of the output image in pixels (width, height)
    
    Returns:
        bytes: The image data for the water edge image
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(blue_water_image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # The color we're looking for is RGB(144,218,238) which is BGR(238,218,144) in OpenCV
    # Create a mask for pixels that are close to this water color
    # Allow some tolerance in the color detection
    lower_bound = np.array([218, 198, 124], dtype=np.uint8)  # BGR with some tolerance
    upper_bound = np.array([255, 238, 164], dtype=np.uint8)  # BGR with some tolerance
    
    # Create the water mask
    water_mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # Find the edges of the water bodies
    edges = cv2.Canny(water_mask, 100, 200)
    
    # Create a black image
    black_image = np.zeros(img.shape, dtype=np.uint8)
    
    # Set the edges to green (BGR format: 0,255,0)
    black_image[edges > 0] = [0, 255, 0]
    
    # Convert back to bytes
    _, buffer = cv2.imencode('.png', black_image)
    return buffer.tobytes()

def crop_and_resize_image(image_data: bytes, target_size: Tuple[int, int]) -> bytes:
    ''' gets rid of the google maps logo/copyright shit at the bottom
    so we can piece the images together '''
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Crop the bottom 40 pixels (where the attribution is)
    height = img.shape[0]
    cropped_img = img[0:height-40, :]
    
    # Resize to target dimensions
    resized_img = cv2.resize(cropped_img, target_size)
    
    # Convert back to bytes
    _, buffer = cv2.imencode('.png', resized_img)
    return buffer.tobytes()

def save_map_image(
    latitude: float,
    longitude: float,
    output_path: str,
    zoom_level: int = 21,  # default is maximum zoom
    api_key: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640),
    maptype: str = 'satellite',
    style_strings: Optional[list] = None):
 
    image_data, metadata = get_map_image(
        latitude=latitude,
        longitude=longitude,
        zoom_level=zoom_level,
        api_key=api_key,
        image_size=image_size,
        maptype=maptype,
        style_strings=style_strings
    )
    
    with open(output_path, 'wb') as f:
        f.write(image_data)
    
    # Print metadata
    print(f"\nImage Metadata ({maptype} map):")
    print(f"Zoom Level: {zoom_level}")
    print(f"Meters per pixel: {metadata['meters_per_pixel']:.2f}")
    print(f"Image width: {metadata['width_meters']:.1f} meters")
    print(f"Image height: {metadata['height_meters']:.1f} meters")
    print(f"Total area covered: {metadata['area_square_meters']:.1f} square meters")
    
    return metadata

def save_water_maps(
    latitude: float,
    longitude: float,
    satellite_output_path: str,
    blue_water_output_path: str,
    water_edge_output_path: str,
    zoom_level: int = 21,
    api_key: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640)):
    """
    Save three map images:
    1. Satellite image
    2. Blue water map with no place names or labels
    3. Water edge map (green edges on black background)
    
    All at the same location and zoom level.
    
    Args:
        latitude (float): Latitude of the center point
        longitude (float): Longitude of the center point
        satellite_output_path (str): Path to save the satellite image
        blue_water_output_path (str): Path to save the blue water map image (without labels)
        water_edge_output_path (str): Path to save the water edge image (green on black)
        zoom_level (int): Zoom level (0-21)
        api_key (str, optional): Google Maps API key
        image_size (tuple): Size of the output images in pixels (width, height)
    """
    # Save satellite image
    satellite_metadata = save_map_image(
        latitude=latitude,
        longitude=longitude,
        zoom_level=zoom_level,
        output_path=satellite_output_path,
        api_key=api_key,
        image_size=image_size,
        maptype='satellite'
    )
    
    # Get blue water map image with no labels
    blue_water_image, blue_water_metadata = get_blue_water_map_image(
        latitude=latitude,
        longitude=longitude,
        zoom_level=zoom_level,
        api_key=api_key,
        image_size=image_size
    )
    
    # Save the blue water image
    with open(blue_water_output_path, 'wb') as f:
        f.write(blue_water_image)
    
    # Create and save the water edge image
    water_edge_image = get_water_edge_image(blue_water_image, image_size)
    with open(water_edge_output_path, 'wb') as f:
        f.write(water_edge_image)
    
    # Print blue water image metadata
    print(f"\nImage Metadata (blue water map without labels):")
    print(f"Zoom Level: {zoom_level}")
    print(f"Meters per pixel: {blue_water_metadata['meters_per_pixel']:.2f}")
    print(f"Image width: {blue_water_metadata['width_meters']:.1f} meters")
    print(f"Image height: {blue_water_metadata['height_meters']:.1f} meters")
    print(f"Total area covered: {blue_water_metadata['area_square_meters']:.1f} square meters")
    
    print(f"\nAll three images saved successfully!")
    return satellite_metadata, blue_water_metadata

def save_both_map_images(
    latitude: float,
    longitude: float,
    satellite_output_path: str,
    blue_water_output_path: str,
    zoom_level: int = 21,
    api_key: Optional[str] = None,
    image_size: Tuple[int, int] = (640, 640)):
    """
    Save both satellite and blue water map images at the same location and zoom level.
    The blue water map will have no place names or labels.
    
    Args:
        latitude (float): Latitude of the center point
        longitude (float): Longitude of the center point
        satellite_output_path (str): Path to save the satellite image
        blue_water_output_path (str): Path to save the blue water map image (without labels)
        zoom_level (int): Zoom level (0-21)
        api_key (str, optional): Google Maps API key
        image_size (tuple): Size of the output images in pixels (width, height)
    """
    # For backward compatibility, call the new function without the water edge output
    water_edge_output_path = "zuggy.png"  # Default path for water edge output
    
    return save_water_maps(
        latitude=latitude,
        longitude=longitude,
        satellite_output_path=satellite_output_path,
        blue_water_output_path=blue_water_output_path,
        water_edge_output_path=water_edge_output_path,
        zoom_level=zoom_level,
        api_key=api_key,
        image_size=image_size
    )

# Example 
if __name__ == "__main__":
    lat, lon = 51.37960940856918, -2.367823990430824
    
    try:
        # Save all three map images using API key from config
        save_water_maps(
            latitude=lat,
            longitude=lon,
            zoom_level=17,
            satellite_output_path="my_house.png",
            blue_water_output_path="alex.png",
            water_edge_output_path="zuggy.png"
        )
    except Exception as e:
        print(f"Error: {e}") 