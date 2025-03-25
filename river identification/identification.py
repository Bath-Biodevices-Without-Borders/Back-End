# use canny edge detection and k means with blue line heuristic thingy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the image
image_path = os.path.join(current_dir, 'river_image.png')
# Get path to estimate image
estimate_path = os.path.join(current_dir, 'estimate.png')
# Load the images using the full paths
image = cv2.imread(image_path)
estimate_image = cv2.imread(estimate_path)

if image is None:
    print(f"Error: Could not read image at path: {image_path}")
    print(f"Check if the file exists at: {os.path.abspath(image_path)}")

if estimate_image is None:
    print(f"Error: Could not read estimate image at path: {estimate_path}")
    print(f"Check if the file exists at: {os.path.abspath(estimate_path)}")


def contouring(cv2_image):
    '''Using canny edge detection and contours to identify a river in a image read by OpenCV
    Input: cv2_image - Image read by OpenCV
    Output: Visualisation of the original image and the image with river contours
    Adjustments: Canny edge detection thresholds, contour settings (CHAIN_APPROX_SIMPLE)
    '''
    #c onvert the image to grayscale
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # change these params
    
    edges = cv2.Canny(blurred, 50, 150) # Can adjust the lower and upper thresholds during testing.
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Retrieves only the external contours as we only
    # want the boundary of the river. Simple is just a setting to save memory

    #draw the contours on the original image to visualize the river
    river_image = cv2_image.copy()
    cv2.drawContours(river_image, contours, -1, (0, 255, 0), 2)  # Green color for the river contour

    # Display the original image and the image with river contours
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(river_image, cv2.COLOR_BGR2RGB)), plt.title('River Contours')
    plt.show()
    

def detect_river_boundaries(cv2_image, show_plots=False):

    # Convert to RGB for processing and visualization
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV color space which is better for segmenting water
    image_hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
    
    # Enhance blue channel which is dominant in water
    blue_channel = cv2_image[:,:,0].copy()  # BGR format, so blue is index 0
    blue_channel = cv2.GaussianBlur(blue_channel, (5, 5), 0)
    
    # Use adaptive thresholding to better segment water
    # Parameters may need adjustment based on the specific image
    thresh_value = cv2.adaptiveThreshold(
        blue_channel, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Also use K-means for a secondary segmentation approach
    pixels = image_hsv.reshape(-1, 3)
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)  # Increase to 4 clusters for better separation
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Identify water cluster (typically has higher S value and lower V value in HSV)
    # This can be improved with more specific water detection heuristics
    s_v_ratio = centroids[:, 1] / (centroids[:, 2] + 1e-10)  # S/V ratio
    river_cluster = np.argmax(s_v_ratio)
    
    # Create kmeans-based mask
    segmented_labels = labels.reshape(image_hsv.shape[:2])
    kmeans_mask = (segmented_labels == river_cluster).astype(np.uint8)
    
    # Combine both masks for better segmentation
    combined_mask = np.logical_or(thresh_value > 127, kmeans_mask > 0).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    river_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    river_mask = cv2.morphologyEx(river_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the boundaries of the river
    # Instead of Canny, we'll use morphological gradient which is better for this task
    gradient_kernel = np.ones((3, 3), np.uint8)
    river_boundaries = cv2.morphologyEx(river_mask, cv2.MORPH_GRADIENT, gradient_kernel)
    
    # Apply Canny edge detection on the mask for sharper boundaries
    canny_edges = cv2.Canny(river_mask * 255, 50, 150)
    
    # Combine both edge detection methods
    combined_edges = np.logical_or(river_boundaries > 0, canny_edges > 0).astype(np.uint8) * 255
    
    # Dilate the edges to make them more visible
    boundary_thickness = 2
    dilated_edges = cv2.dilate(combined_edges, np.ones((boundary_thickness, boundary_thickness), np.uint8))
    
    # Create colored overlay for visualization
    boundary_overlay = image_rgb.copy()
    boundary_overlay[dilated_edges > 0] = [255, 0, 0]  # Red boundaries
    
    # Create a translucent version with original image
    alpha = 0.7
    blended_image = cv2.addWeighted(image_rgb, alpha, boundary_overlay, 1-alpha, 0)
    
    river_colored = image_rgb.copy()
    river_area = river_mask > 0
    # Blue tint for river
    river_colored[river_area] = (river_colored[river_area] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
    
    # Final image with blue river and red boundaries
    final_image = river_colored.copy()
    final_image[dilated_edges > 0] = [255, 0, 0]  # Red boundaries
    
    if show_plots:
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # River mask
        plt.subplot(232)
        plt.imshow(river_mask * 255, cmap='gray')
        plt.title('River Segmentation')
        plt.axis('off')
        
        # Edge detection result
        plt.subplot(233)
        plt.imshow(dilated_edges, cmap='gray')
        plt.title('River Boundaries')
        plt.axis('off')
        
        # Colored river area
        plt.subplot(234)
        plt.imshow(river_colored)
        plt.title('River Area')
        plt.axis('off')
        
        # Boundaries overlaid on original image
        plt.subplot(235)
        plt.imshow(blended_image)
        plt.title('Boundaries Overlay')
        plt.axis('off')
        
        # Final visualization with colored river and boundaries
        plt.subplot(236)
        plt.imshow(final_image)
        plt.title('Final Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Return the images for potential further processing
    return {
        'original': image_rgb,
        'mask': river_mask,
        'boundaries': dilated_edges,
        'overlay': blended_image,
        'final': final_image
    }


def guided_river_boundaries(cv2_image, guide_image, show_plots=False, save_plots=False):
    # guide image is the estimate.png image with green lines marking the river boundaries
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    guide_rgb = cv2.cvtColor(guide_image, cv2.COLOR_BGR2RGB)
    
    # Extract green channel from guide image (where boundaries are marked)
    # green channel is [1] in RGB
    green_channel = guide_rgb[:, :, 1]
    
    # Create a mask from the guide image (green lines only)
    _, guide_mask = cv2.threshold(green_channel, 150, 255, cv2.THRESH_BINARY)
    
   
    kernel = np.ones((3, 3), np.uint8)
    guide_mask_dilated = cv2.dilate(guide_mask, kernel, iterations=3)  # Increased iterations for wider guide
    
    # Run the normal river detection to get candidate boundaries
    detection_results = detect_river_boundaries(cv2_image, show_plots=False)  # Don't show plots from this step
    river_mask = detection_results['mask']
    detected_boundaries = detection_results['boundaries']
    

    buffer_size = 25  # increase buffer size to capture more of the river area
    guide_roi = cv2.dilate(guide_mask_dilated, np.ones((buffer_size, buffer_size), np.uint8))
    
    # Filter the detected boundaries using the guide ROI
    # Only keep boundaries that are close to the estimated boundaries
    filtered_boundaries = cv2.bitwise_and(detected_boundaries, guide_roi)
    
    #  refine the river mask by finding the area enclosed by the filtered boundaries
    # create a solid mask by filling in the area within the boundaries
    filled_mask = np.zeros_like(filtered_boundaries)
    contours, _ = cv2.findContours(filtered_boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Only process contours that are large enough to be significant
    significant_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small noise contours
            significant_contours.append(contour)
    
    cv2.drawContours(filled_mask, significant_contours, -1, 255, -1)  # -1 fills the contour
    
    # Apply morphological operations to fill gaps and smooth the mask
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Combine with original river mask for a better result
    refined_river_mask = cv2.bitwise_and(river_mask * 255, filled_mask)
    
    # Apply closing operation to fill small gaps in the mask
    refined_river_mask = cv2.morphologyEx(refined_river_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # Find the boundaries of the refined river area
    refined_boundaries = cv2.Canny(refined_river_mask, 50, 150)
    # Dilate the boundaries to make them more visible
    refined_boundaries = cv2.dilate(refined_boundaries, np.ones((2, 2), np.uint8))
    
    # Create visualization - WITHOUT blue shading for the river area
    # Only show the boundaries overlaid on the original image
    final_image = image_rgb.copy()
    final_image[refined_boundaries > 0] = [255, 0, 0]  # Red boundaries only
    
    # Create a translucent overlay of boundaries on original image
    boundary_overlay = image_rgb.copy()
    boundary_overlay[refined_boundaries > 0] = [255, 0, 0]  # Red boundaries
    alpha = 0.7
    blended_image = cv2.addWeighted(image_rgb, alpha, boundary_overlay, 1-alpha, 0)
    
    # Create a comparison image with original and guide overlaid
    guide_overlay = image_rgb.copy()
    guide_contours, _ = cv2.findContours(guide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(guide_overlay, guide_contours, -1, (0, 255, 0), 2)  # Green for guide
    
    # Create a combined image with both original and filtered boundaries
    combined_boundaries = image_rgb.copy()
    combined_boundaries[detected_boundaries > 0] = [0, 255, 255]  # Yellow for original
    combined_boundaries[refined_boundaries > 0] = [255, 0, 0]  # Red for refined
    
    if show_plots or save_plots:
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # Guide mask
        plt.subplot(232)
        plt.imshow(guide_overlay)
        plt.title('Guide Boundaries (Green)')
        plt.axis('off')
        
        # Combined boundaries
        plt.subplot(233)
        plt.imshow(combined_boundaries)
        plt.title('All vs Filtered Boundaries')
        plt.axis('off')
        
        # Filtered boundaries
        plt.subplot(234)
        plt.imshow(filtered_boundaries, cmap='gray')
        plt.title('Filtered Boundaries')
        plt.axis('off')
        
        # Refined river mask
        plt.subplot(235)
        plt.imshow(refined_river_mask, cmap='gray')
        plt.title('Refined River Area')
        plt.axis('off')
        
        # Final visualization
        plt.subplot(236)
        plt.imshow(final_image)
        plt.title('Final Result (Boundaries Only)')
        plt.axis('off')
        
        plt.tight_layout()
       
        if save_plots:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(current_dir, 'guided_river_result.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {output_path}")
            
        
            cv2.imwrite(os.path.join(current_dir, 'final_river_boundaries.png'), 
                        cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(current_dir, 'filtered_boundaries.png'), 
                        filtered_boundaries)
            cv2.imwrite(os.path.join(current_dir, 'guide_overlay.png'),
                        cv2.cvtColor(guide_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(current_dir, 'combined_boundaries.png'),
                        cv2.cvtColor(combined_boundaries, cv2.COLOR_RGB2BGR))
        
        if show_plots:
            plt.show()
    
    return {
        'original': image_rgb,
        'guide_mask': guide_mask,
        'filtered_boundaries': filtered_boundaries,
        'refined_mask': refined_river_mask,
        'refined_boundaries': refined_boundaries,
        'final': final_image
    }


if __name__ == "__main__":
    if image is not None and estimate_image is not None:
        print("Running guided river boundary detection...")
        guided_river_boundaries(image, estimate_image, show_plots=True, save_plots=True)
        
    else:
        print("Cannot run analysis functions: Image(s) failed to load.")
    