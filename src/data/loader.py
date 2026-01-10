import os
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from torch.utils.data import Dataset
import torch
from src.simulation.weather_models import FogSimulator, RainSimulator, SnowSimulator

def collate_fn(batch):
    return tuple(zip(*batch))

class NuScenesWeatherDataset(Dataset):
    def __init__(self, root_dir, version='v1.0-mini', split='train', transform=None):
        self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=False)
        self.root_dir = root_dir
        self.transform = transform
        
        # Weather Simulators
        self.fog = FogSimulator()
        self.rain = RainSimulator()
        self.snow = SnowSimulator()
        self.current_severity = 0.0 # Default clear
        self.weather_mode = 'random'
        
        # Get scenes and build sample list
        # Simplified for now: just grab all keyframes
        # Filter samples by split
        splits = create_splits_scenes()
        
        # Handle split name mapping if needed (e.g. user passes 'train' but version is mini -> 'mini_train')
        if version == 'v1.0-mini':
            if split == 'train': split = 'mini_train'
            if split == 'val': split = 'mini_val'
            
        if split not in splits:
            raise ValueError(f"Available splits: {list(splits.keys())}")
            
        scene_names = splits[split]
        
        self.samples = []
        for s in self.nusc.sample:
            scene_token = s['scene_token']
            scene_name = self.nusc.get('scene', scene_token)['name']
            if scene_name in scene_names:
                self.samples.append(s)
                
        print(f"Found {len(self.samples)} samples for split {split} (Version: {version})")
        
    def set_weather_severity(self, severity, mode='random'):
        self.current_severity = severity
        self.weather_mode = mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get Camera Data
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # Load Image
        img_path = os.path.join(self.root_dir, cam_data['filename'])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Depth Map (Required for Weather)
        if self.current_severity > 0:
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            depth_map = self._get_depth_map(cam_data, lidar_data)
            
            # Apply Weather
            if self.weather_mode == 'random':
                 w_type = np.random.choice(['fog', 'rain', 'snow'])
            else:
                 w_type = self.weather_mode

            if w_type == 'fog':
                 # --- NEW PyTorch Guided Filter Fog ---
                 # Visibility: Low severity (0.1) -> 800m? High (1.0) -> 50m
                 # Map severity [0,1] to visibility [800, 30]
                 vis_m = 800.0 - (self.current_severity * 750.0) 
                 vis_m = max(30.0, vis_m)
                 
                 from src.simulation.fog_torch import FogGenerator
                 fog_gen = FogGenerator(visibility_m=vis_m)
                 
                 # Prepare Tensors (B, C, H, W)
                 img_t = torch.from_numpy(image_np).permute(2,0,1).float().unsqueeze(0) / 255.0
                 d_t = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0)
                 
                 # Apply
                 with torch.no_grad():
                     aug_t = fog_gen(img_t, d_t)
                     
                 # Convert back for downstream compatibility or update image_np
                 # Since rest of pipeline expects image_np for Rain/Snow logic or tensor later
                 # But we are in "if fog" block.
                 # Let's overwrite image_np
                 aug_img = aug_t.squeeze(0).permute(1,2,0).numpy() * 255.0
                 image_np = np.clip(aug_img, 0, 255).astype(np.uint8)
                 
            elif w_type == 'rain':
                 image_np = self.rain.add_rain(image_np, depth_map, rainfall_rate=self.current_severity)
            elif w_type == 'snow':
                 image_np = self.snow.add_snow(image_np, depth_map, snow_rate=self.current_severity)

        # Convert to Tensor (0-1 float)
        # Faster R-CNN expects List[Tensor[C, H, W]] in 0-1 range
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        # Get Annotations
        # Faster R-CNN target format:
        # - boxes (FloatTensor[N, 4]): [x1, y1, x2, y2]
        # - labels (Int64Tensor[N]): class label
        
        _, boxes_obj, _ = self.nusc.get_sample_data(cam_token)
        
        boxes = []
        labels = []
        
        for box in boxes_obj:
            # Map categories to 1-10 IDs (Simple mapping)
            # NuScenes has 'human.pedestrian.adult', 'vehicle.car', etc.
            # Simplified: 1=human, 2=vehicle, 3=others
            cls_name = box.name
            label_id = 1 # default
            if 'human' in cls_name: label_id = 1
            elif 'vehicle' in cls_name: label_id = 2
            else: label_id = 3
            
            # Convert box to 2D [x1, y1, x2, y2]
            # Box is in 3D. We need to project corners to 2D
            # NuScenes box.corners() returns 3x8
            # This is complex. Use existing API tools ideally.
            # view_points again.
            
            # For simplicity in this research framework step:
            # We assume we can get 2D boxes or project them.
            # Implementing projection logic for bounding boxes:
            
            corners_3d = box.corners()
            corners_img = view_points(corners_3d, np.array(self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic']), normalize=True)[:2, :]
            
            x_min = np.min(corners_img[0, :])
            x_max = np.max(corners_img[0, :])
            y_min = np.min(corners_img[1, :])
            y_max = np.max(corners_img[1, :])
            
            # Clip to image size
            H, W = 900, 1600
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(W, x_max)
            y_max = min(H, y_max)
            
            if x_max > x_min and y_max > y_min:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(label_id)

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Negative sample (no objects)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
            
        target["image_id"] = torch.tensor([idx])

        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, target

    def _get_depth_map(self, cam_data, lidar_data):
        """
        Projects LiDAR point cloud to camera image plane.
        """
        # Load Point Cloud
        pcl_path = os.path.join(self.root_dir, lidar_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        
        from pyquaternion import Quaternion

        # Transform Points: Lidar -> Global -> Ego -> Cam
        # 1. Lidar -> Ego
        cs_record_lidar = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        # Quaternion to Rotation Matrix
        lidar_rot = Quaternion(cs_record_lidar['rotation']).rotation_matrix
        pc.rotate(lidar_rot)
        pc.translate(np.array(cs_record_lidar['translation']))
        
        # 2. Ego -> Global (at lidar timestamp) -> Global (at cam timestamp) -> Ego (at cam timestamp)
        # To simplify: We assume ego motion is compensated or negligible for simple projection
        # Ideally: transform from lidar_pose to cam_pose via global
        poserecord_lidar = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        poserecord_cam = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Lidar Ego -> Global
        ego_lidar_rot = Quaternion(poserecord_lidar['rotation']).rotation_matrix
        pc.rotate(ego_lidar_rot)
        pc.translate(np.array(poserecord_lidar['translation']))
        
        # Global -> Cam Ego
        pc.translate(-np.array(poserecord_cam['translation']))
        ego_cam_rot = Quaternion(poserecord_cam['rotation']).rotation_matrix
        pc.rotate(ego_cam_rot.T)
        
        # 3. Cam Ego -> Cam Image
        cs_record_cam = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_rot = Quaternion(cs_record_cam['rotation']).rotation_matrix
        pc.translate(-np.array(cs_record_cam['translation']))
        pc.rotate(cam_rot.T)
        
        # Filter points outside camera view (z > 0)
        # depth = points[2, :]
        
        points = view_points(pc.points[:3, :], np.array(cs_record_cam['camera_intrinsic']), normalize=True)
        
        # Create Sparse Depth Map
        H, W = 900, 1600 # NuScenes image dimensions
        depth_map = np.zeros((H, W), dtype=np.float32)
        
        # Points is 3xN: [u, v, z] (after view_points z is dropped? No, view_points returns 3xN)
        # Actually view_points returns 3xN where row 0 is u, row 1 is v, row 2 is usually 1.0 (homogeneous) or depth?
        # Check docs: view_points projects to 2d. 
        # We need the depth Z from the camera frame coordinates *before* projection.
        
        # Re-doing projection carefully to keep Z
        # Coordinate of point in camera frame:
        # P_cam = R_inv * (P_global - T) ... done above steps
        # Z is P_cam[2]
        
        # We grabbed coordinates from pc.points before `view_points`.
        # pc.points is (4, N)
        depths = pc.points[2, :]
        
        # Remove points behind camera
        mask = depths > 0.1
        points = points[:, mask]
        depths = depths[mask]
        
        # Remove points outside image plane
        u_mask = (points[0, :] >= 0) & (points[0, :] < W)
        v_mask = (points[1, :] >= 0) & (points[1, :] < H)
        mask_uv = u_mask & v_mask
        
        points = points[:, mask_uv]
        depths = depths[mask_uv]
        
        # Fill depth map
        # Optimization: use integer coordinates
        u = points[0, :].astype(np.int32)
        v = points[1, :].astype(np.int32)
        
        # Handle scaling issues? NuScenes is 1600x900
        # If multiple points hit same pixel, take min depth (closest)
        # Simple for-loop is slow, use scatter logic or simply assign
        # Since this is "Offline" or "batch" generation, efficiency is key but correctness first.
        
        # Simple assignment (overwrites, acceptable for sparse)
        depth_map[v, u] = depths
        
        # Dilation to fill holes (approximate dense depth)
        # This is a heuristic needed for fog - pure sparse fog looks spotty
        # Dilation to fill holes (approximate dense depth)
        import cv2
        # Use a larger kernel to bridge lidar rings
        kernel = np.ones((5,5), np.uint8)
        
        # 1. Morphological Closing to fill intra-object holes
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 2. Dilate to spread depth to nearby pixels
        dense_depth_map = cv2.dilate(depth_map, kernel, iterations=3)
        
        # 3. Handle Missing Values (Sky vs Holes)
        # Any pixel still 0.0 is likely sky or far distance (beyond lidar range)
        # We set them to a "Max Visibility" distance (e.g. 1000m)
        # This ensures they get fully fogged.
        
        # However, purely setting all 0s to 1000 might look weird on edges.
        # Ideally we perform simple inpainting.
        
        # Mask of valid pixels
        mask = dense_depth_map > 0
        
        if np.sum(mask) == 0:
             # Fallback if no lidar points projected
             return np.full((H, W), 100.0, dtype=np.float32)

        # Simple Nearest Neighbor Inpainting (via distance transform)
        # Invert mask (0 where valid, 1 where invalid)
        # On second thought, simply setting zeros to a large value (Sky) 
        # is the standard hack for sparse lidar IF we assume holes on objects are closed.
        
        dense_depth_map[dense_depth_map == 0] = 300.0 # Set infinite/sky depth
        
        return dense_depth_map

    def get_raw_sample(self, idx):
        """
        Returns raw numpy image and depth map for visualization.
        """
        sample = self.samples[idx]
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        
        # Image
        img_path = os.path.join(self.root_dir, cam_data['filename'])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Depth
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        depth_map = self._get_depth_map(cam_data, lidar_data)
        
        return {"image": image_np, "depth": depth_map}
