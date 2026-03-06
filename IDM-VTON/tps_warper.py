"""
FitFusion — Thin Plate Spline (TPS) Warping Module
====================================================
Pre-warps the cloth image towards the target body shape using
landmark correspondences estimated from body segmentation.

This provides an explicit geometric conditioning signal to the UNet,
improving garment placement accuracy especially for size-varying try-ons.

Usage in training:
    warper = TPSWarper()
    warped_cloth = warper(cloth_image, source_landmarks, target_landmarks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TPSWarper(nn.Module):
    """
    Thin Plate Spline warping using control point correspondences.
    
    Given source and target control points, computes a smooth deformation
    field and applies it to warp the cloth image towards the target body shape.
    """
    
    def __init__(self, num_control_points: int = 16):
        super().__init__()
        self.num_cp = num_control_points
    
    def _compute_tps_kernel(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute TPS radial basis kernel matrix K.
        
        Args:
            points: (N, 2) control points
        Returns:
            K: (N, N) kernel matrix
        """
        N = points.shape[0]
        diff = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 2)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        # TPS kernel: U(r) = r^2 * log(r)
        # Avoid log(0) by adding small epsilon
        dist = dist.clamp(min=1e-6)
        K = dist ** 2 * torch.log(dist)
        return K
    
    def _solve_tps(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve for TPS transformation parameters.
        
        Args:
            source_points: (N, 2) source control points
            target_points: (N, 2) target control points
        Returns:
            params: (N+3, 2) TPS parameters [weights; affine]
        """
        N = source_points.shape[0]
        device = source_points.device
        dtype = source_points.dtype
        
        K = self._compute_tps_kernel(source_points)  # (N, N)
        
        # Build P matrix: [1, x, y] for each control point
        P = torch.cat([
            torch.ones(N, 1, device=device, dtype=dtype),
            source_points,
        ], dim=1)  # (N, 3)
        
        # Build system matrix
        # [K  P] [w]   [target]
        # [P' 0] [a] = [  0   ]
        top = torch.cat([K, P], dim=1)  # (N, N+3)
        bot = torch.cat([P.T, torch.zeros(3, 3, device=device, dtype=dtype)], dim=1)  # (3, N+3)
        L = torch.cat([top, bot], dim=0)  # (N+3, N+3)
        
        # RHS
        rhs = torch.cat([
            target_points,
            torch.zeros(3, 2, device=device, dtype=dtype),
        ], dim=0)  # (N+3, 2)
        
        # Solve
        params = torch.linalg.solve(L + 1e-4 * torch.eye(N + 3, device=device, dtype=dtype), rhs)
        return params
    
    def _apply_tps(
        self,
        params: torch.Tensor,
        source_points: torch.Tensor,
        grid_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply TPS transformation to grid points.
        
        Args:
            params: (N+3, 2) TPS parameters
            source_points: (N, 2) source control points
            grid_points: (H*W, 2) points to transform
        Returns:
            transformed: (H*W, 2) transformed points
        """
        N = source_points.shape[0]
        weights = params[:N]     # (N, 2)
        affine = params[N:]      # (3, 2)
        
        # Compute kernel values between grid and source
        diff = grid_points.unsqueeze(1) - source_points.unsqueeze(0)  # (H*W, N, 2)
        dist = torch.norm(diff, dim=-1).clamp(min=1e-6)  # (H*W, N)
        K = dist ** 2 * torch.log(dist)  # (H*W, N)
        
        # Affine part: a0 + a1*x + a2*y
        P = torch.cat([
            torch.ones(grid_points.shape[0], 1, device=grid_points.device, dtype=grid_points.dtype),
            grid_points,
        ], dim=1)  # (H*W, 3)
        
        # Transformed = K @ weights + P @ affine
        transformed = K @ weights + P @ affine
        return transformed
    
    def forward(
        self,
        cloth: torch.Tensor,
        source_landmarks: torch.Tensor,
        target_landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp cloth image from source shape to target shape.
        
        Args:
            cloth: (B, C, H, W) cloth image tensor
            source_landmarks: (B, N, 2) source control points (normalized -1 to 1)
            target_landmarks: (B, N, 2) target control points (normalized -1 to 1)
        
        Returns:
            warped: (B, C, H, W) warped cloth image
        """
        B, C, H, W = cloth.shape
        device = cloth.device
        dtype = cloth.dtype
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
        
        warped_list = []
        for b in range(B):
            # Solve TPS
            params = self._solve_tps(
                target_landmarks[b].to(dtype),
                source_landmarks[b].to(dtype),
            )
            
            # Apply to get sampling coordinates
            sampling_coords = self._apply_tps(params, target_landmarks[b].to(dtype), grid)
            sampling_grid = sampling_coords.reshape(1, H, W, 2)
            
            # Sample
            warped = F.grid_sample(
                cloth[b:b+1],
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            warped_list.append(warped)
        
        return torch.cat(warped_list, dim=0)


def estimate_body_landmarks(
    seg_map: np.ndarray,
    num_points: int = 16,
) -> np.ndarray:
    """
    Estimate body control points from a segmentation map.
    
    Returns control points at key body positions (shoulders, waist, hips, etc.)
    normalized to [-1, 1].
    
    Args:
        seg_map: (H, W) segmentation label map
        num_points: number of control points to generate
    
    Returns:
        landmarks: (num_points, 2) normalized control points
    """
    H, W = seg_map.shape
    
    # Find bounding box of body (non-background)
    body_mask = seg_map > 0
    if not body_mask.any():
        # Return evenly spaced grid as fallback
        ys = np.linspace(-0.8, 0.8, int(np.sqrt(num_points)))
        xs = np.linspace(-0.4, 0.4, int(np.sqrt(num_points)))
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        return grid[:num_points]
    
    ys, xs = np.where(body_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # Generate landmarks at key body positions
    landmarks = []
    y_steps = np.linspace(y_min, y_max, num_points // 2 + 2)[1:-1]
    
    for y in y_steps:
        y_int = int(y)
        row = body_mask[max(0, y_int-5):min(H, y_int+5), :]
        if row.any():
            row_xs = np.where(row.any(axis=0))[0]
            if len(row_xs) >= 2:
                landmarks.append([row_xs.min(), y])  # Left edge
                landmarks.append([row_xs.max(), y])  # Right edge
    
    landmarks = np.array(landmarks[:num_points])
    
    # Pad if not enough points
    while len(landmarks) < num_points:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        landmarks = np.vstack([landmarks, [cx, cy]])
    
    # Normalize to [-1, 1]
    landmarks[:, 0] = landmarks[:, 0] / W * 2 - 1
    landmarks[:, 1] = landmarks[:, 1] / H * 2 - 1
    
    return landmarks[:num_points]
