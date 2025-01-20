import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocesing import KITTIDataLoader  # Import from Stage 1

class DepthEstimator:
    def __init__(self, base_path):
        """
        Initialize the depth estimation pipeline
        Args:
            base_path (str): Path to the KITTI dataset root directory
        """
        self.data_loader = KITTIDataLoader(base_path)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,  # must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11 ** 2,  # First parameter controlling disparity smoothness
            P2=32 * 3 * 11 ** 2,  # Second parameter controlling disparity smoothness
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        # Get camera parameters from calibration
        self.P_rect_00 = self.data_loader.calibration.get('P_rect_00', None)
        self.P_rect_01 = self.data_loader.calibration.get('P_rect_01', None)
        
        if self.P_rect_00 is not None and self.P_rect_01 is not None:
            # Reshape projection matrices
            self.P_rect_00 = self.P_rect_00.reshape(3, 4)
            self.P_rect_01 = self.P_rect_01.reshape(3, 4)
            
            # Extract focal length and baseline
            self.focal_length = self.P_rect_00[0, 0]  # Focal length in pixels
            self.baseline = abs(self.P_rect_01[0, 3] / self.P_rect_01[0, 0])  # Baseline in meters
        else:
            self.focal_length = None
            self.baseline = None

    def preprocess_images(self, left_img, right_img):
        """
        Preprocess stereo images for better disparity estimation
        Args:
            left_img (np.array): Left stereo image
            right_img (np.array): Right stereo image
        Returns:
            tuple: Preprocessed left and right images
        """
        # Ensure images are grayscale
        if len(left_img.shape) > 2:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        if len(right_img.shape) > 2:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
        # Apply histogram equalization
        left_img = cv2.equalizeHist(left_img)
        right_img = cv2.equalizeHist(right_img)
        
        return left_img, right_img

    def compute_disparity(self, left_img, right_img):
        """
        Compute disparity map from stereo images
        Args:
            left_img (np.array): Left stereo image
            right_img (np.array): Right stereo image
        Returns:
            np.array: Disparity map
        """
        # Preprocess images
        left_processed, right_processed = self.preprocess_images(left_img, right_img)
        
        # Compute disparity
        disparity = self.stereo.compute(left_processed, right_processed).astype(np.float32) / 16.0
        
        return disparity

    def disparity_to_depth(self, disparity):
        """
        Convert disparity map to depth map
        Args:
            disparity (np.array): Disparity map
        Returns:
            np.array: Depth map in meters
        """
        # Avoid division by zero
        mask = disparity > 0
        depth = np.zeros_like(disparity)
        
        # Convert disparity to depth using focal length and baseline
        if self.focal_length is not None and self.baseline is not None:
            depth[mask] = (self.focal_length * self.baseline) / disparity[mask]
        else:
            print("Warning: Calibration parameters not available")
            
        # Clip depth to reasonable range (0-80 meters for KITTI)
        depth = np.clip(depth, 0, 80)
        
        return depth

    def process_frame(self, frame_idx):
        """
        Process a single frame to generate depth map
        Args:
            frame_idx (int): Frame index to process
        Returns:
            tuple: Original left image, disparity map, and depth map
        """
        # Load stereo pair
        frame_data = self.data_loader.load_frame(frame_idx)
        left_img = frame_data['gray_left']
        right_img = frame_data['gray_right']
        
        # Compute disparity
        disparity = self.compute_disparity(left_img, right_img)
        
        # Convert to depth
        depth = self.disparity_to_depth(disparity)
        
        return left_img, disparity, depth

    def visualize_results(self, left_img, disparity, depth):
        """
        Visualize the original image, disparity map, and depth map
        Args:
            left_img (np.array): Original left image
            disparity (np.array): Computed disparity map
            depth (np.array): Computed depth map
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.title('Original Left Image')
        plt.imshow(left_img, cmap='gray')
        plt.axis('off')
        
        # Disparity map
        plt.subplot(132)
        plt.title('Disparity Map')
        plt.imshow(disparity, cmap='jet')
        plt.colorbar(label='Disparity')
        plt.axis('off')
        
        # Depth map
        plt.subplot(133)
        plt.title('Depth Map')
        plt.imshow(depth, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def test_depth_estimation():
    """Test function to verify the depth estimation pipeline"""
    base_path = r"C:\Users\nihar\Documents\github\Automated_path_planning\data"
    depth_estimator = DepthEstimator(base_path)
    
    # Process and visualize a test frame
    frame_idx = 0
    left_img, disparity, depth = depth_estimator.process_frame(frame_idx)
    depth_estimator.visualize_results(left_img, disparity, depth)
    
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: {depth.min():.2f}m to {depth.max():.2f}m")

if __name__ == "__main__":
    test_depth_estimation()