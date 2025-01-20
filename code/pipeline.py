import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import logging
from pathlib import Path

from preprocesing import KITTIDataLoader
from depth_estimator import DepthEstimator
from obstacle_detector import ObstacleDetector
from cost_map import CostMapGenerator
from path import PathPlanner

class IntegratedPipeline:
    def __init__(self, base_path):
        """
        Initialize the integrated pipeline
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.base_path = base_path
        self.data_loader = KITTIDataLoader(base_path)
        self.depth_estimator = DepthEstimator(base_path)
        self.obstacle_detector = ObstacleDetector(base_path)
        self.cost_map_generator = CostMapGenerator(base_path)
        self.path_planner = PathPlanner(base_path)
        
        # Performance metrics
        self.metrics = {
            'data_loading': [],
            'depth_estimation': [],
            'obstacle_detection': [],
            'cost_map': [],
            'path_planning': [],
            'total': []
        }
    
    def process_frame(self, frame_idx, start_pos=None, goal_pos=None):
        """
        Process a single frame through the entire pipeline
        Args:
            frame_idx (int): Frame index to process
            start_pos (tuple): Start position for path planning (optional)
            goal_pos (tuple): Goal position for path planning (optional)
        Returns:
            dict: Dictionary containing all intermediate and final results
        """
        results = {}
        total_start = time()
        
        try:
            # Stage 1: Data Loading
            stage_start = time()
            frame_data = self.data_loader.load_frame(frame_idx)
            results['frame_data'] = frame_data
            self.metrics['data_loading'].append(time() - stage_start)
            self.logger.info(f"Stage 1 complete - Frame {frame_idx}")
            
            # Stage 2: Depth Estimation
            stage_start = time()
            left_img, disparity, depth_map = self.depth_estimator.process_frame(frame_idx)
            results['depth_data'] = {
                'left_img': left_img,
                'disparity': disparity,
                'depth_map': depth_map
            }
            self.metrics['depth_estimation'].append(time() - stage_start)
            self.logger.info("Stage 2 complete - Depth estimation")
            
            # Stage 3: Obstacle Detection
            stage_start = time()
            rgb_image, obstacle_mask, obstacles, positions = self.obstacle_detector.process_frame(frame_idx)
            results['obstacle_data'] = {
                'rgb_image': rgb_image,
                'obstacle_mask': obstacle_mask,
                'obstacles': obstacles,
                'positions': positions
            }
            self.metrics['obstacle_detection'].append(time() - stage_start)
            self.logger.info("Stage 3 complete - Obstacle detection")
            
            # Stage 4: Cost Map Generation
            stage_start = time()
            rgb_image, occupancy_grid, cost_map = self.cost_map_generator.process_frame(frame_idx)
            results['cost_map_data'] = {
                'rgb_image': rgb_image,
                'occupancy_grid': occupancy_grid,
                'cost_map': cost_map
            }
            self.metrics['cost_map'].append(time() - stage_start)
            self.logger.info("Stage 4 complete - Cost map generation")
            
            # Stage 5: Path Planning
            stage_start = time()
            rgb_image, cost_map, path = self.path_planner.process_frame(frame_idx, start_pos, goal_pos)
            results['path_data'] = {
                'rgb_image': rgb_image,
                'cost_map': cost_map,
                'path': path
            }
            self.metrics['path_planning'].append(time() - stage_start)
            self.logger.info("Stage 5 complete - Path planning")
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_idx}: {str(e)}")
            raise
        
        self.metrics['total'].append(time() - total_start)
        return results
    
    def visualize_complete_pipeline(self, results):
        """
        Visualize all stages of the pipeline
        Args:
            results (dict): Results from process_frame
        """
        plt.figure(figsize=(20, 10))
        
        # Original images (Stage 1)
        plt.subplot(231)
        plt.title('Stereo Images')
        plt.imshow(cv2.cvtColor(results['frame_data']['rgb_left'], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Depth visualization (Stage 2)
        plt.subplot(232)
        plt.title('Depth Map')
        plt.imshow(results['depth_data']['depth_map'], cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.axis('off')
        
        # Obstacle detection (Stage 3)
        plt.subplot(233)
        plt.title('Obstacle Detection')
        obstacle_vis = results['obstacle_data']['rgb_image'].copy()
        for box in results['obstacle_data']['obstacles']:
            x, y, w, h = box['bbox']
            cv2.rectangle(obstacle_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(obstacle_vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Cost map (Stage 4)
        plt.subplot(234)
        plt.title('Cost Map')
        plt.imshow(results['cost_map_data']['cost_map'], cmap='YlOrRd')
        plt.colorbar(label='Cost')
        plt.axis('off')
        
        # Path planning (Stage 5)
        plt.subplot(235)
        plt.title('Planned Path')
        path_vis = results['cost_map_data']['cost_map'].copy()
        if results['path_data']['path']:
            path = np.array(results['path_data']['path'])
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='Path')
        plt.imshow(path_vis, cmap='YlOrRd')
        plt.axis('off')
        
        # Performance metrics
        plt.subplot(236)
        plt.title('Performance Metrics')
        stages = ['Data\nLoading', 'Depth\nEst.', 'Obstacle\nDet.', 
                 'Cost\nMap', 'Path\nPlan.', 'Total']
        times = [self.metrics[k][-1] for k in ['data_loading', 'depth_estimation',
                'obstacle_detection', 'cost_map', 'path_planning', 'total']]
        plt.bar(stages, times)
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.show()
    
    def run_tests(self, n_frames=10):
        """
        Run comprehensive tests on multiple frames
        Args:
            n_frames (int): Number of frames to test
        """
        self.logger.info(f"Starting comprehensive testing on {n_frames} frames")
        
        all_results = []
        for i in range(n_frames):
            try:
                results = self.process_frame(i)
                all_results.append(results)
                self.logger.info(f"Successfully processed frame {i}")
            except Exception as e:
                self.logger.error(f"Failed to process frame {i}: {str(e)}")
        
        # Calculate and display statistics
        print("\nPerformance Statistics:")
        for stage, times in self.metrics.items():
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"{stage.replace('_', ' ').title()}:")
            print(f"  Mean: {mean_time:.3f}s")
            print(f"  Std:  {std_time:.3f}s")
            
        # Save test results
        save_path = Path(self.base_path) / "test_results"
        save_path.mkdir(exist_ok=True)
        
        # Save performance metrics
        np.save(save_path / "performance_metrics.npy", self.metrics)
        self.logger.info(f"Test results saved to {save_path}")

def test_integrated_pipeline():
    """Test function to verify the complete pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    pipeline = IntegratedPipeline(base_path)
    
    # Process single frame
    frame_idx = 0
    results = pipeline.process_frame(frame_idx)
    pipeline.visualize_complete_pipeline(results)
    
    # Run comprehensive tests
    pipeline.run_tests(n_frames=5)

if __name__ == "__main__":
    test_integrated_pipeline()