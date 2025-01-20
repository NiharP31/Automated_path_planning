import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
import logging
from pathlib import Path
from tqdm import tqdm

# Import previous stages
from preprocesing import KITTIDataLoader
from depth_estimator import DepthEstimator
from obstacle_detector import ObstacleDetector
from cost_map import CostMapGenerator
from path import PathPlanner
from pipeline import IntegratedPipeline



class VideoIntegrationPipeline:
    def __init__(self, base_path, output_path):
        """
        Initialize the video processing pipeline
        Args:
            base_path (str): Path to the KITTI dataset root directory
            output_path (str): Path to save output video
        """
        self.base_path = base_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.integrated_pipeline = IntegratedPipeline(base_path)
        
        # Video writer parameters
        self.frame_size = (1920, 480)  # Will show 4 views side by side
        self.fps = 10
        self.video_writer = None
        
    def initialize_video_writer(self):
        """Initialize video writer with defined parameters"""
        output_file = str(self.output_path / "processed_sequence.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_file, fourcc, self.fps, self.frame_size
        )
        
    def create_visualization_frame(self, results):
        """
        Create a single visualization frame combining all results.

        Args:
            results (dict): Results from pipeline processing.

        Returns:
            np.array: Combined visualization frame.
        """
        # Original RGB image with obstacle detection
        rgb_with_obstacles = results['obstacle_data']['rgb_image'].copy()
        for box in results['obstacle_data']['obstacles']:
            x, y, w, h = box['bbox']
            cv2.rectangle(rgb_with_obstacles, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Depth map visualization
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(
                results['depth_data']['depth_map'], 
                alpha=255 / 50
            ), 
            cv2.COLORMAP_VIRIDIS
        )

        # Cost map visualization with path
        cost_map_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(
                results['cost_map_data']['cost_map'] * 255,
                alpha=1.0
            ),
            cv2.COLORMAP_HOT
        )
        if results['path_data']['path']:
            path = np.array(results['path_data']['path'])
            for i in range(len(path) - 1):
                pt1 = (int(path[i][1]), int(path[i][0]))
                pt2 = (int(path[i+1][1]), int(path[i+1][0]))
                cv2.line(cost_map_vis, pt1, pt2, (0, 255, 0), 2)

        # Top-down view visualization
        top_down = np.zeros_like(rgb_with_obstacles)
        for pos in results['obstacle_data']['positions']:
            x, z = int(pos[0] * 10 + top_down.shape[1] / 2), int(pos[2] * 10)
            cv2.circle(top_down, (x, z), 5, (0, 0, 255), -1)

        # Resize visualizations to ensure consistent height for horizontal stacking (row 2)
        target_height_row2 = max(cost_map_vis.shape[0], top_down.shape[0])
        cost_map_vis = cv2.resize(cost_map_vis, (cost_map_vis.shape[1], target_height_row2))
        top_down = cv2.resize(top_down, (top_down.shape[1], target_height_row2))

        # Combine cost map and top-down views horizontally
        vis_row2 = np.hstack([cost_map_vis, top_down])

        # Resize visualizations to ensure consistent height for horizontal stacking (row 1)
        target_height_row1 = max(rgb_with_obstacles.shape[0], depth_vis.shape[0])
        rgb_with_obstacles = cv2.resize(rgb_with_obstacles, (rgb_with_obstacles.shape[1], target_height_row1))
        depth_vis = cv2.resize(depth_vis, (depth_vis.shape[1], target_height_row1))

        # Combine RGB and depth visualizations horizontally
        vis_row1 = np.hstack([rgb_with_obstacles, depth_vis])

        # Ensure both rows have the same width for vertical stacking
        target_width = max(vis_row1.shape[1], vis_row2.shape[1])
        vis_row1 = cv2.resize(vis_row1, (target_width, vis_row1.shape[0]))
        vis_row2 = cv2.resize(vis_row2, (target_width, vis_row2.shape[0]))

        # Combine rows vertically
        combined_frame = np.vstack([vis_row1, vis_row2])

        # Resize to desired frame size
        combined_frame = cv2.resize(combined_frame, self.frame_size)

        return combined_frame




        
    def process_sequence(self, start_frame=0, num_frames=None):
        """
        Process a sequence of frames and create video
        Args:
            start_frame (int): Starting frame index
            num_frames (int): Number of frames to process (None for all)
        """
        # Initialize video writer
        self.initialize_video_writer()
        
        # Get total number of frames
        total_frames = len(self.integrated_pipeline.data_loader.image_lists['gray_left'])
        if num_frames is None:
            num_frames = total_frames - start_frame
        
        # Process frames
        try:
            for frame_idx in tqdm(range(start_frame, start_frame + num_frames)):
                if frame_idx >= total_frames:
                    break
                    
                # Process frame
                results = self.integrated_pipeline.process_frame(frame_idx)
                
                # Create visualization
                vis_frame = self.create_visualization_frame(results)
                
                # Write to video
                self.video_writer.write(vis_frame)
                
            # Release video writer
            self.video_writer.release()
            logging.info(f"Video saved to {self.output_path}")
            
        except Exception as e:
            logging.error(f"Error processing sequence: {str(e)}")
            if self.video_writer is not None:
                self.video_writer.release()
            raise
            
    def process_with_temporal_consistency(self, start_frame=0, num_frames=None,
                                        smoothing_window=5):
        """
        Process sequence with temporal smoothing
        Args:
            start_frame (int): Starting frame index
            num_frames (int): Number of frames to process
            smoothing_window (int): Window size for temporal smoothing
        """
        self.initialize_video_writer()
        
        # Initialize temporal buffers
        depth_buffer = []
        obstacle_buffer = []
        path_buffer = []
        
        total_frames = len(self.integrated_pipeline.data_loader.image_lists['gray_left'])
        if num_frames is None:
            num_frames = total_frames - start_frame
            
        try:
            for frame_idx in tqdm(range(start_frame, start_frame + num_frames)):
                if frame_idx >= total_frames:
                    break
                    
                # Process current frame
                results = self.integrated_pipeline.process_frame(frame_idx)
                
                # Update temporal buffers
                depth_buffer.append(results['depth_data']['depth_map'])
                obstacle_buffer.append(results['obstacle_data']['positions'])
                if results['path_data']['path']:
                    path_buffer.append(results['path_data']['path'])
                    
                # Apply temporal smoothing
                if len(depth_buffer) > smoothing_window:
                    depth_buffer.pop(0)
                    obstacle_buffer.pop(0)
                    if len(path_buffer) > smoothing_window:
                        path_buffer.pop(0)
                        
                # Smooth depth map
                results['depth_data']['depth_map'] = np.mean(depth_buffer, axis=0)
                
                # Smooth obstacle positions
                if len(obstacle_buffer) > 1:
                    smoothed_positions = []
                    for i in range(len(obstacle_buffer[-1])):
                        positions = [buf[i] for buf in obstacle_buffer 
                                   if i < len(buf)]
                        if positions:
                            smoothed_positions.append(
                                tuple(np.mean(positions, axis=0))
                            )
                    results['obstacle_data']['positions'] = smoothed_positions
                
                # Smooth path
                if path_buffer:
                    smoothed_path = []
                    max_len = max(len(p) for p in path_buffer)
                    for i in range(max_len):
                        points = [p[i] for p in path_buffer if i < len(p)]
                        if points:
                            smoothed_path.append(
                                tuple(np.mean(points, axis=0))
                            )
                    results['path_data']['path'] = smoothed_path
                
                # Create and write visualization
                vis_frame = self.create_visualization_frame(results)
                self.video_writer.write(vis_frame)
                
            self.video_writer.release()
            logging.info(f"Video with temporal consistency saved to {self.output_path}")
            
        except Exception as e:
            logging.error(f"Error processing sequence: {str(e)}")
            if self.video_writer is not None:
                self.video_writer.release()
            raise

def test_video_pipeline():
    """Test function to verify the video processing pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    output_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\output"
    
    pipeline = VideoIntegrationPipeline(base_path, output_path)
    
    # Process a short sequence without temporal consistency
    pipeline.process_sequence(start_frame=0, num_frames=446)
    
    # Process with temporal consistency
    pipeline.process_with_temporal_consistency(start_frame=0, num_frames=446)

if __name__ == "__main__":
    test_video_pipeline()