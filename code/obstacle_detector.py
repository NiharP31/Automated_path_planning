import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocesing import KITTIDataLoader
from depth_estimator import DepthEstimator

class ObstacleDetector:
    def __init__(self, base_path):
        """
        Initialize the obstacle detection system
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        self.data_loader = KITTIDataLoader(base_path)
        self.depth_estimator = DepthEstimator(base_path)
        
        # Initialize object detection parameters
        self.min_contour_area = 500  # Increased minimum area for obstacle detection
        self.depth_threshold = 50.0  # Maximum depth to consider for obstacles (meters)
        self.min_depth = 2.0  # Minimum depth to consider (to filter out noise)
        
    def detect_ground_plane(self, depth_map):
        """
        Detect and remove ground plane from depth map
        Args:
            depth_map (np.array): Input depth map
        Returns:
            np.array: Binary mask with ground plane removed
        """
        rows, cols = depth_map.shape
        ground_mask = np.zeros_like(depth_map, dtype=np.uint8)
        
        # Assume lower 1/3 of image contains ground
        horizon_line = int(2 * rows / 3)
        
        # Create gradient in y-direction
        gradient_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        
        # Ground typically has smooth depth transition
        ground_mask[horizon_line:, :] = 1
        gradient_threshold = np.std(gradient_y) * 2
        ground_mask[np.abs(gradient_y) > gradient_threshold] = 0
        
        return ground_mask
    
    def detect_obstacles(self, depth_map, rgb_image):
        """
        Detect obstacles using depth map and RGB image
        Args:
            depth_map (np.array): Depth map from stereo estimation
            rgb_image (np.array): RGB image for visualization
        Returns:
            tuple: Binary obstacle mask and list of obstacle bounding boxes
        """
        # Remove ground plane
        ground_mask = self.detect_ground_plane(depth_map)
        depth_mask = np.logical_and(depth_map > 0, depth_map < self.depth_threshold)
        
        # Create obstacle mask
        obstacle_mask = np.logical_and(depth_mask, ground_mask == 0).astype(np.uint8) * 255
        
        # Find contours of potential obstacles
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        obstacle_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                mean_depth = np.mean(depth_map[y:y+h, x:x+w][depth_map[y:y+h, x:x+w] > 0])
                obstacle_boxes.append({
                    'bbox': (x, y, w, h),
                    'depth': mean_depth,
                    'area': area
                })
        
        return obstacle_mask, obstacle_boxes
    
    def project_to_3d(self, obstacle_boxes, depth_map):
        """
        Project detected obstacles to 3D space
        Args:
            obstacle_boxes (list): List of detected obstacle bounding boxes
            depth_map (np.array): Depth map for 3D projection
        Returns:
            list: 3D positions of obstacles (x, y, z in meters)
        """
        obstacle_positions = []
        focal_length = self.depth_estimator.focal_length
        
        if focal_length is None:
            return obstacle_positions
            
        for obstacle in obstacle_boxes:
            x, y, w, h = obstacle['bbox']
            depth = obstacle['depth']
            
            # Calculate 3D position (assuming camera center as origin)
            center_x = x + w/2
            center_y = y + h/2
            
            X = (center_x - depth_map.shape[1]/2) * depth / focal_length
            Y = (center_y - depth_map.shape[0]/2) * depth / focal_length
            Z = depth
            
            obstacle_positions.append((X, Y, Z))
            
        return obstacle_positions
    
    def process_frame(self, frame_idx):
        """
        Process a single frame for obstacle detection
        Args:
            frame_idx (int): Frame index to process
        Returns:
            tuple: Original RGB image, obstacle mask, obstacle boxes, and 3D positions
        """
        # Get frame data
        frame_data = self.data_loader.load_frame(frame_idx)
        rgb_image = frame_data['rgb_left']
        
        # Get depth map
        _, _, depth_map = self.depth_estimator.process_frame(frame_idx)
        
        # Detect obstacles
        obstacle_mask, obstacle_boxes = self.detect_obstacles(depth_map, rgb_image)
        
        # Project to 3D
        obstacle_positions = self.project_to_3d(obstacle_boxes, depth_map)
        
        return rgb_image, obstacle_mask, obstacle_boxes, obstacle_positions
    
    def visualize_obstacles(self, rgb_image, obstacle_boxes, obstacle_positions):
        """
        Visualize detected obstacles
        Args:
            rgb_image (np.array): Original RGB image
            obstacle_boxes (list): List of detected obstacle bounding boxes
            obstacle_positions (list): List of 3D positions of obstacles
        """
        # Create copy for visualization
        vis_image = rgb_image.copy()
        
        # Draw bounding boxes and depths
        for box, position in zip(obstacle_boxes, obstacle_positions):
            x, y, w, h = box['bbox']
            depth = box['depth']
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw depth and 3D position
            label = f"D:{depth:.1f}m\nX:{position[0]:.1f}m\nZ:{position[2]:.1f}m"
            y_offset = y
            for line in label.split('\n'):
                y_offset += 20
                cv2.putText(vis_image, line, (x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.title('Obstacle Detection')
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Plot top-down view of obstacles
        plt.subplot(122)
        plt.title('Top-down View (X-Z plane)')
        plt.scatter([p[0] for p in obstacle_positions],
                   [p[2] for p in obstacle_positions],
                   c='r', marker='o')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def test_obstacle_detection():
    """Test function to verify the obstacle detection pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    detector = ObstacleDetector(base_path)
    
    # Process and visualize a test frame
    frame_idx = 0
    rgb_image, obstacle_mask, obstacle_boxes, obstacle_positions = detector.process_frame(frame_idx)
    detector.visualize_obstacles(rgb_image, obstacle_boxes, obstacle_positions)
    
    print(f"Number of obstacles detected: {len(obstacle_boxes)}")
    print("Obstacle positions (X, Y, Z meters):")
    for pos in obstacle_positions:
        print(f"  {pos}")

if __name__ == "__main__":
    test_obstacle_detection()