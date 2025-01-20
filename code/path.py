import numpy as np
import cv2
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from preprocesing import KITTIDataLoader
from cost_map import CostMapGenerator

class PathPlanner:
    def __init__(self, base_path):
        """
        Initialize the path planner
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        self.cost_map_generator = CostMapGenerator(base_path)
        
        # Path planning parameters
        self.movement_cost = 1.0
        self.cost_weight = 10.0  # Weight for cost map values
        self.heuristic_weight = 1.2  # Weight for heuristic (> 1 for faster planning)
        
        # Define possible movements (8-connected grid)
        self.movements = [(-1, -1), (-1, 0), (-1, 1),
                         (0, -1),          (0, 1),
                         (1, -1),  (1, 0),  (1, 1)]
        self.movement_costs = [1.414, 1.0, 1.414,
                             1.0,         1.0,
                             1.414, 1.0, 1.414]  # Diagonal cost is √2
    
    def heuristic(self, a, b):
        """
        Calculate heuristic distance between two points
        Args:
            a (tuple): First point (x, y)
            b (tuple): Second point (x, y)
        Returns:
            float: Estimated distance
        """
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    def get_neighbors(self, current, cost_map):
        """
        Get valid neighboring cells
        Args:
            current (tuple): Current cell coordinates
            cost_map (np.array): Cost map
        Returns:
            list: List of valid neighbors and their costs
        """
        neighbors = []
        rows, cols = cost_map.shape
        
        for move, base_cost in zip(self.movements, self.movement_costs):
            next_x = current[0] + move[0]
            next_y = current[1] + move[1]
            
            # Check bounds
            if 0 <= next_x < rows and 0 <= next_y < cols:
                # Check if cell is traversable (cost < 0.8)
                if cost_map[next_x, next_y] < 0.8:
                    # Total cost includes movement cost and cost map value
                    total_cost = (base_cost + 
                                self.cost_weight * cost_map[next_x, next_y])
                    neighbors.append(((next_x, next_y), total_cost))
        
        return neighbors
    
    def plan_path(self, cost_map, start, goal):
        """
        Plan path using A* algorithm
        Args:
            cost_map (np.array): Cost map
            start (tuple): Start position (x, y)
            goal (tuple): Goal position (x, y)
        Returns:
            list: Path as list of positions
        """
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor, cost in self.get_neighbors(current, cost_map):
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = (tentative_g + 
                                       self.heuristic_weight * 
                                       self.heuristic(neighbor, goal))
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def optimize_path(self, path, cost_map):
        """
        Optimize path using path smoothing
        Args:
            path (list): Original path
            cost_map (np.array): Cost map
        Returns:
            list: Smoothed path
        """
        if not path or len(path) < 3:
            return path
        
        smooth_path = [path[0]]
        current_point = 0
        
        while current_point < len(path) - 1:
            # Look ahead for furthest visible point
            for i in range(len(path)-1, current_point, -1):
                if self.check_line_of_sight(path[current_point], 
                                          path[i], 
                                          cost_map):
                    smooth_path.append(path[i])
                    current_point = i
                    break
                    
        return smooth_path
    
    def check_line_of_sight(self, start, end, cost_map):
        """
        Check if there's a clear line of sight between two points
        Args:
            start (tuple): Start point
            end (tuple): End point
            cost_map (np.array): Cost map
        Returns:
            bool: True if there's clear line of sight
        """
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x = x0
        y = y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            if cost_map[int(x), int(y)] >= 0.8:
                return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
        return True
    
    def process_frame(self, frame_idx, start_pos=None, goal_pos=None):
        """
        Process a single frame for path planning
        Args:
            frame_idx (int): Frame index to process
            start_pos (tuple): Start position (optional)
            goal_pos (tuple): Goal position (optional)
        Returns:
            tuple: RGB image, cost map, and planned path
        """
        # Get cost map
        rgb_image, _, cost_map = self.cost_map_generator.process_frame(frame_idx)
        
        # Set default start and goal if not provided
        if start_pos is None:
            start_pos = (cost_map.shape[0]-10, cost_map.shape[1]//2)
        if goal_pos is None:
            goal_pos = (10, cost_map.shape[1]//2)
        
        # Plan and optimize path
        path = self.plan_path(cost_map, start_pos, goal_pos)
        if path:
            path = self.optimize_path(path, cost_map)
        
        return rgb_image, cost_map, path
    
    def visualize_path(self, rgb_image, cost_map, path):
        """
        Visualize the planned path
        Args:
            rgb_image (np.array): Original RGB image
            cost_map (np.array): Cost map
            path (list): Planned path
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Cost map with path
        plt.subplot(132)
        plt.title('Cost Map with Path')
        plt.imshow(cost_map, cmap='YlOrRd')
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_y, path_x, 'b-', linewidth=2, label='Planned Path')
        plt.colorbar(label='Cost')
        plt.axis('off')
        
        # Zoomed cost map with path
        plt.subplot(133)
        plt.title('Zoomed Path View')
        if path:
            # Calculate zoom region
            path_x = np.array([p[0] for p in path])
            path_y = np.array([p[1] for p in path])
            center_x = (path_x.max() + path_x.min()) // 2
            center_y = (path_y.max() + path_y.min()) // 2
            zoom_size = max(path_x.max() - path_x.min(),
                          path_y.max() - path_y.min()) + 20
            
            plt.imshow(cost_map[max(0, center_x-zoom_size//2):
                               min(cost_map.shape[0], center_x+zoom_size//2),
                               max(0, center_y-zoom_size//2):
                               min(cost_map.shape[1], center_y+zoom_size//2)],
                      cmap='YlOrRd')
            plt.plot(path_y - max(0, center_y-zoom_size//2),
                    path_x - max(0, center_x-zoom_size//2),
                    'b-', linewidth=2)
        plt.colorbar(label='Cost')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test_path_planning():
    """Test function to verify the path planning pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    planner = PathPlanner(base_path)
    
    # Process and visualize a test frame
    frame_idx = 350
    rgb_image, cost_map, path = planner.process_frame(frame_idx)
    planner.visualize_path(rgb_image, cost_map, path)
    
    if path:
        print(f"Path found with {len(path)} waypoints")
        print("Path length:", sum(np.sqrt((path[i][0]-path[i-1][0])**2 +
                                        (path[i][1]-path[i-1][1])**2)
                                for i in range(1, len(path))))
    else:
        print("No path found")

if __name__ == "__main__":
    test_path_planning()