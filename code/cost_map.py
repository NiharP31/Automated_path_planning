import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocesing import KITTIDataLoader
from depth_estimator import DepthEstimator
from obstacle_detector import ObstacleDetector

class CostMapGenerator:
    def __init__(self, base_path):
        """
        Initialize the cost map generator
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        self.data_loader = KITTIDataLoader(base_path)
        self.depth_estimator = DepthEstimator(base_path)
        self.obstacle_detector = ObstacleDetector(base_path)
        
        # Cost map parameters
        self.resolution = 0.2  # meters per pixel
        self.map_size = (50, 50)  # meters
        self.grid_size = (int(self.map_size[0]/self.resolution), 
                         int(self.map_size[1]/self.resolution))
        
    def create_occupancy_grid(self, obstacle_positions):
        """
        Create binary occupancy grid from obstacle positions
        Args:
            obstacle_positions (list): List of (x, y, z) obstacle positions
        Returns:
            np.array: Binary occupancy grid
        """
        grid = np.zeros(self.grid_size, dtype=np.uint8)
        
        # Convert to grid coordinates
        center_x = self.grid_size[0] // 2
        center_y = self.grid_size[1] // 2
        
        for x, _, z in obstacle_positions:
            # Convert world coordinates to grid coordinates
            grid_x = int(center_x + x / self.resolution)
            grid_y = int(center_y + z / self.resolution)
            
            # Check if within grid bounds
            if (0 <= grid_x < self.grid_size[0] and 
                0 <= grid_y < self.grid_size[1]):
                # Add obstacle with inflation
                cv2.circle(grid, (grid_x, grid_y), 
                          int(1.0/self.resolution),  # 1 meter radius
                          1, -1)
        
        return grid
    
    def create_cost_map(self, occupancy_grid, depth_map):
        """
        Create cost map combining occupancy grid and depth information
        Args:
            occupancy_grid (np.array): Binary occupancy grid
            depth_map (np.array): Depth map from stereo estimation
        Returns:
            np.array: Cost map
        """
        cost_map = np.zeros(self.grid_size, dtype=np.float32)
        
        # Add costs from occupancy grid (obstacles)
        cost_map += occupancy_grid * 0.7  # High cost for obstacles
        
        # Add distance transform for obstacle proximity
        dist_transform = cv2.distanceTransform(1 - occupancy_grid, 
                                             cv2.DIST_L2, 3)
        max_distance = 5.0 / self.resolution  # 5 meters
        proximity_cost = np.clip(1 - dist_transform / max_distance, 0, 1) * 0.3
        
        cost_map += proximity_cost
        
        # Normalize cost map
        cost_map = np.clip(cost_map, 0, 1)
        
        return cost_map
        
    def process_frame(self, frame_idx):
        """
        Process a single frame to generate cost map
        Args:
            frame_idx (int): Frame index to process
        Returns:
            tuple: RGB image, occupancy grid, and cost map
        """
        # Get RGB image and depth map
        frame_data = self.data_loader.load_frame(frame_idx)
        rgb_image = frame_data['rgb_left']
        _, _, depth_map = self.depth_estimator.process_frame(frame_idx)
        
        # Get obstacle positions
        _, _, _, obstacle_positions = self.obstacle_detector.process_frame(frame_idx)
        
        # Create occupancy grid
        occupancy_grid = self.create_occupancy_grid(obstacle_positions)
        
        # Generate cost map
        cost_map = self.create_cost_map(occupancy_grid, depth_map)
        
        return rgb_image, occupancy_grid, cost_map
    
    def visualize_maps(self, rgb_image, occupancy_grid, cost_map):
        """
        Visualize the original image, occupancy grid, and cost map
        Args:
            rgb_image (np.array): Original RGB image
            occupancy_grid (np.array): Binary occupancy grid
            cost_map (np.array): Generated cost map
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Occupancy grid
        plt.subplot(132)
        plt.title('Occupancy Grid')
        plt.imshow(occupancy_grid, cmap='binary')
        plt.colorbar(label='Occupancy')
        plt.axis('off')
        
        # Cost map
        plt.subplot(133)
        plt.title('Cost Map')
        plt.imshow(cost_map, cmap='YlOrRd')
        plt.colorbar(label='Cost')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test_cost_map():
    """Test function to verify the cost map generation pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    cost_map_gen = CostMapGenerator(base_path)
    
    # Process and visualize a test frame
    frame_idx = 0
    rgb_image, occupancy_grid, cost_map = cost_map_gen.process_frame(frame_idx)
    cost_map_gen.visualize_maps(rgb_image, occupancy_grid, cost_map)
    
    print(f"Occupancy grid shape: {occupancy_grid.shape}")
    print(f"Cost map shape: {cost_map.shape}")
    print(f"Cost range: {cost_map.min():.2f} to {cost_map.max():.2f}")

if __name__ == "__main__":
    test_cost_map()