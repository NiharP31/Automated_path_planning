import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class KITTIDataLoader:
    def __init__(self, base_path):
        """
        Initialize the KITTI data loader
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        self.base_path = base_path
        self.image_folders = {
            'gray_left': os.path.join(base_path, '2011_09_26_drive_0009_sync/image_00/data'),
            'gray_right': os.path.join(base_path, '2011_09_26_drive_0009_sync/image_01/data'),
            'rgb_left': os.path.join(base_path, '2011_09_26_drive_0009_sync/image_02/data'),
            'rgb_right': os.path.join(base_path, '2011_09_26_drive_0009_sync/image_03/data')
        }
        
        # Load calibration data
        self.calib_path = os.path.join(base_path, '2011_09_26_calib')
        self.calibration = self._load_calibration()
        
        # Get sorted image lists
        self.image_lists = self._get_image_lists()
        
    def _load_calibration(self):
        """Load and parse calibration files"""
        calib_data = {}
        
        # Load camera-to-camera calibration
        cam_to_cam_path = os.path.join(self.calib_path, 'calib_cam_to_cam.txt')
        if os.path.exists(cam_to_cam_path):
            with open(cam_to_cam_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    key, value = line.split(':', 1)
                    try:
                        # Convert string to numpy array for calibration matrices
                        calib_data[key] = np.array([float(x) for x in value.strip().split()])
                    except:
                        calib_data[key] = value.strip()
        
        return calib_data
    
    def _get_image_lists(self):
        """Get sorted lists of images for each camera"""
        image_lists = {}
        for key, folder in self.image_folders.items():
            if os.path.exists(folder):
                images = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
                image_lists[key] = images
        return image_lists
    
    def load_frame(self, frame_idx):
        """
        Load synchronized images from all cameras for a given frame index
        Args:
            frame_idx (int): Frame index to load
        Returns:
            dict: Dictionary containing the synchronized images
        """
        if frame_idx >= len(self.image_lists['gray_left']):
            raise IndexError(f"Frame index {frame_idx} out of range")
            
        frame_data = {}
        for key in self.image_folders.keys():
            img_path = os.path.join(self.image_folders[key], self.image_lists[key][frame_idx])
            if 'rgb' in key:
                frame_data[key] = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else:
                frame_data[key] = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
        return frame_data
    
    def visualize_frame(self, frame_data):
        """
        Visualize the loaded frame data
        Args:
            frame_data (dict): Dictionary containing the synchronized images
        """
        plt.figure(figsize=(15, 10))
        
        # Plot grayscale images
        plt.subplot(2, 2, 1)
        plt.title('Left Grayscale')
        plt.imshow(frame_data['gray_left'], cmap='gray')
        
        plt.subplot(2, 2, 2)
        plt.title('Right Grayscale')
        plt.imshow(frame_data['gray_right'], cmap='gray')
        
        # Plot RGB images
        plt.subplot(2, 2, 3)
        plt.title('Left RGB')
        plt.imshow(cv2.cvtColor(frame_data['rgb_left'], cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 2, 4)
        plt.title('Right RGB')
        plt.imshow(cv2.cvtColor(frame_data['rgb_right'], cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        plt.show()

def test_data_loader():
    """Test function to verify the data loader"""
    # Initialize data loader
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"  # Update this path
    loader = KITTIDataLoader(base_path)
    
    # Load and visualize a test frame
    test_frame = loader.load_frame(0)
    loader.visualize_frame(test_frame)
    
    print("Number of frames available:", len(loader.image_lists['gray_left']))
    print("Calibration data keys:", loader.calibration.keys())

if __name__ == "__main__":
    test_data_loader()