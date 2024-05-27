"""
Density proposal field
"""
from typing import Optional, Callable
import logging as log

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.models.kplane_field import interpolate_ms_features, normalize_aabb, init_grid_param
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


class KPlaneDensityField(nn.Module):
    def __init__(self,
                 aabb,
                 resolution,
                 num_input_coords,
                 num_output_coords,
                 density_activation: Callable,
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 linear_decoder: bool = True):
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.hexplane = num_input_coords == 4
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        activation = "ReLU"
        if self.linear_decoder:
            activation = "None"

        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution,
            a=0.1, b=0.15)
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        log.info(f"Initialized KPlaneDensityField. hexplane={self.hexplane} - "
                 f"resolution={resolution}")
        log.info(f"KPlaneDensityField grids: \n{self.grids}")

    def get_density_cont(self, pts: torch.Tensor):
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]

        # 1st iter
        timestamps = torch.arange(-1.0, 1.0, 1/16).to(pts.device)
        # pts: [8129, 256, 3]
        timestamps = timestamps.unsqueeze(-1).expand(-1, n_rays * n_samples).view(-1, n_rays, n_samples, 1)  # [n_timestamps, n_rays, n_samples, 1]
        pts = pts.unsqueeze(0).expand(timestamps.shape[0], -1, -1, -1)
        input_pts = pts
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_timestamps, n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(*timestamps.shape)

        # 2nd iter
        indices = density.argmax(dim=0, keepdim=True)   # [1, n_rays, n_samples, 1]
        max_timestamps = timestamps.gather(0, indices)  # [1, n_rays, n_samples, 1]
        neighbors = torch.arange(-1/16, 1/16, 1/256).to(indices.device)
        timestamps = neighbors.view(-1, 1, 1, 1) + max_timestamps

        pts = torch.cat((input_pts, timestamps), dim=-1)  # [n_timestamps, n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(*timestamps.shape).max(dim=0).values
        
        return density

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        if timestamps is not None and isinstance(timestamps, int):
            return self.get_density_cont(pts)
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None and self.hexplane:
            # print(timestamps.shape)
            # timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            timestamps = timestamps.unsqueeze(-1).expand(*timestamps.shape, n_samples).unsqueeze(-1)  # [n_timestamps, n_rays, n_samples, 1]
            target_shape = timestamps.shape
            if len(pts.shape) < len(timestamps.shape):
                pts = pts.unsqueeze(0).expand(timestamps.shape[0], -1, -1, -1)
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False, num_levels=None)
        density = self.density_activation(
            self.sigma_net(features).to(pts)
            #features.to(pts)
        ).view(*target_shape)
        if len(target_shape) == 4:    # shape-time rendering
            density = density.max(dim=0).values
        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = {k: v for k, v in self.sigma_net.named_parameters(prefix="sigma_net")}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }
