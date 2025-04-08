import copy
from typing import Any

import numpy as np
import open3d as o3d
import torch

from engine.data_structures import GaussianSplattingData


def recenter_gs_points(points: np.ndarray) -> Any:
    """Function for recentering Gaussian Splatting centroids"""

    recentered_points = (points - points.mean()).astype(np.float32)
    return recentered_points


def transform_gs_data(gs_data: GaussianSplattingData, ref_bbox_size: float = 1.5) -> GaussianSplattingData:
    """Function for rescaling the model to the fixed bbox size"""

    gs_data_out = copy.deepcopy(gs_data)

    points = gs_data.points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox = bbox.create_from_points(pcd.points)
    extent = np.array(bbox.get_extent())

    max_size = np.max(extent)
    scaling = ref_bbox_size / max_size

    centered_points = recenter_gs_points(points)
    gs_data_out.points = torch.tensor(centered_points, dtype=torch.float32) * scaling
    gs_data_out.scales *= scaling
    volume_scale = np.prod(scaling)
    gs_data_out.opacities = torch.clip(gs_data.opacities * (1.0 / volume_scale), 0.0, 1.0).to(torch.float32)

    return gs_data_out
