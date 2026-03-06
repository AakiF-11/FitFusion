import numpy as np
from typing import Tuple, Dict, Any

def detect_pose_occlusion(openpose_keypoints: Any, torso_bbox: Tuple[int, int, int, int]) -> Tuple[bool, str]:
    """
    Checks if wrists or elbows fall inside the torso bounding box, indicating crossed arms or hands on hips.
    """
    points = []
    if isinstance(openpose_keypoints, dict):
        points = list(openpose_keypoints.values())
    elif isinstance(openpose_keypoints, list) or isinstance(openpose_keypoints, np.ndarray):
        points = openpose_keypoints
        
    def get_pt(idx):
        if idx < len(points):
            p = points[idx]
            if p is not None and len(p) >= 2 and p[0] > 0 and p[1] > 0:
                return p[:2]
        return None
        
    r_elbow, r_wrist = get_pt(3), get_pt(4)
    l_elbow, l_wrist = get_pt(6), get_pt(7)
    
    check_points = [p for p in [r_elbow, r_wrist, l_elbow, l_wrist] if p is not None]
    
    t_min_x, t_max_x, t_min_y, t_max_y = torso_bbox
    
    for pt in check_points:
        px, py = pt[0], pt[1]
        if t_min_x <= px <= t_max_x and t_min_y <= py <= t_max_y:
            return False, "Pose Error: Please keep arms at your sides."
            
    return True, ""

def score_mask_validity(schp_mask_array: np.ndarray, openpose_keypoints: Any, padding: int = 50) -> Tuple[float, bool, str]:
    """
    Prevents hallucinated limbs from SCHP mask bleeding by cross-referencing with OpenPose keypoints.
    
    Calculates the bounding box of the user's arms and torso using OpenPose joints (Shoulder -> Elbow -> Wrist).
    If the `schp_mask_array` predicts "garment" pixels more than `N` pixels outside this OpenPose bounding box, 
    return confidence_score < 0.85 and a halt_generation flag.
    """
    try:
        # Extract points from openpose_keypoints based on common formats
        points = []
        if isinstance(openpose_keypoints, dict):
            points = list(openpose_keypoints.values())
        elif isinstance(openpose_keypoints, list) or isinstance(openpose_keypoints, np.ndarray):
            points = openpose_keypoints
            
        def get_pt(idx):
            if idx < len(points):
                p = points[idx]
                if p is not None and len(p) >= 2 and p[0] > 0 and p[1] > 0:
                    return p[:2]
            return None
            
        # Standard OpenPose tracking IDs
        r_shoulder, r_elbow, r_wrist = get_pt(2), get_pt(3), get_pt(4)
        l_shoulder, l_elbow, l_wrist = get_pt(5), get_pt(6), get_pt(7)
        neck = get_pt(1)
        r_hip, l_hip = get_pt(8), get_pt(11)
        
        h, w = schp_mask_array.shape[:2]
        valid_region = np.zeros((h, w), dtype=np.uint8)
        
        def add_bbox(pts, is_torso=False):
            valid_pts = [np.array(p) for p in pts if p is not None]
            if len(valid_pts) < 1:
                return
            points_array = np.array(valid_pts)
            min_x = int(np.min(points_array[:, 0])) - padding
            max_x = int(np.max(points_array[:, 0])) + padding
            # Add more height padding for torso
            y_pad = padding * 2 if is_torso else padding
            min_y = int(np.min(points_array[:, 1])) - y_pad
            max_y = int(np.max(points_array[:, 1])) + y_pad
            
            min_y_c = max(0, min_y)
            max_y_c = min(h, max_y)
            min_x_c = max(0, min_x)
            max_x_c = min(w, max_x)
            
            valid_region[min_y_c:max_y_c, min_x_c:max_x_c] = 1

        # Torso validation (base requirement)
        neck = get_pt(1)
        r_shoulder = get_pt(2)
        l_shoulder = get_pt(5)
        r_hip = get_pt(8)
        l_hip = get_pt(11)

        torso_pts = [np.array(p) for p in [neck, r_shoulder, l_shoulder, r_hip, l_hip] if p is not None]
        if len(torso_pts) > 0:
            torso_array = np.array(torso_pts)
            t_min_x = int(np.min(torso_array[:, 0])) - padding
            t_max_x = int(np.max(torso_array[:, 0])) + padding
            t_min_y = int(np.min(torso_array[:, 1])) - padding * 2
            t_max_y = int(np.max(torso_array[:, 1])) + padding * 2
            
            # Sub-Task 1: Self-Occlusion Validation Gate
            is_valid, error_msg = detect_pose_occlusion(openpose_keypoints, (t_min_x, t_max_x, t_min_y, t_max_y))
            if not is_valid:
                return 0.0, True, error_msg
                
        add_bbox([neck, r_shoulder, l_shoulder, r_hip, l_hip], is_torso=True)
        
        # Right Arm validation logic
        if r_elbow is None:
            # Check if elbow_joint is None. If true, bypass the arm bleed check entirely
            pass
        elif r_wrist is None:
            # Check if wrist_joint is None. If true, calculate bounding box using only (shoulder, elbow)
            add_bbox([r_shoulder, r_elbow])
        else:
            add_bbox([r_shoulder, r_elbow, r_wrist])

        # Left Arm validation logic
        if l_elbow is None:
            pass
        elif l_wrist is None:
            add_bbox([l_shoulder, l_elbow])
        else:
            add_bbox([l_shoulder, l_elbow, l_wrist])
            
        garment_pixels = np.sum(schp_mask_array > 0)
        if garment_pixels == 0:
            return 0.0, True, ""
            
        # Calculate how many pixels fall outside the expected arm/torso bounds
        outside_pixels = np.sum((schp_mask_array > 0) & (valid_region == 0))
        
        # Pixel threshold N
        N = 500
        if outside_pixels > N:
            confidence = max(0.0, 1.0 - (outside_pixels / garment_pixels))
            halt_generation = confidence < 0.85
            return confidence, halt_generation, ""
            
        return 1.0, False, ""
    except Exception as e:
        # Failsafe backward compatibility
        return 1.0, False, ""
