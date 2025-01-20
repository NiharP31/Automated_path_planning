# Automated Path Planning System using KITTI Dataset

## Introduction
This project implements an automated path planning system using the KITTI dataset. The system processes stereo camera inputs to create depth maps, detect obstacles, generate cost maps, and plan optimal navigation paths for autonomous vehicles.

## Demo Video
[Demo Video Link - Coming Soon]

Outputs:
- Stereo image processing
- Depth estimation maps
- Obstacle detection and tracking
- Path planning visualization

## Components
![System Flow](flow_diagram.png)

The system consists of six main components:

1. **Data Pipeline**
   - Handles KITTI dataset loading and preprocessing
   - Manages stereo images and calibration data

2. **Depth Estimation**
   - Generates depth maps from stereo images
   - Provides 3D scene understanding

3. **Obstacle Detection**
   - Identifies and tracks obstacles
   - Projects obstacles to 3D space

4. **Cost Map Generation**
   - Creates navigation cost maps
   - Integrates obstacle information

5. **Path Planning**
   - Plans optimal navigation paths
   - Avoids detected obstacles

6. **Integration Pipeline**
   - Combines all components
   - Processes video sequences

## Running Individual Components

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/automated_path_planning.git

# Create virtual environment
conda create -n ENV
conda activate ENV

# Install requirements
pip install -r requirements.txt
```

### Running Components
```python
# Data Pipeline
python code/preprocessing.py

# Depth Estimation
python code/depth_estimator.py

# Obstacle Detection
python code/obstacle_detector.py

# Cost Map Generation
python code/cost_map.py

# Path Planning
python code/path.py

# Full Pipeline
python code/pipeline.py

# Video Generation
python code/video.py
```

## Acknowledgments
- KITTI Dataset for providing autonomous driving data