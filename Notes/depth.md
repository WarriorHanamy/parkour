# Depth Processing Pipeline Notes

This document summarizes the depth image processing pipelines for both real-world deployment and IsaacGym simulation.

## 1. Real-World Depth Processing

### Primary Files

| Robot | File Path | ROS Version |
|-------|-----------|-------------|
| Go2 | `onboard_codes/go2/go2_visual.py` | ROS2 |
| Go1/A1 | `onboard_codes/go1/go1_visual_embedding.py` | ROS1 |

### Real-World Pipeline (Go2)

| Step | Code Location | Function |
|------|---------------|----------|
| 1. Capture | `go2_visual.py:188-192` | `get_depth_frame()` - Acquires depth frame from pyrealsense2 pipeline |
| 2. Realsense Filters | `go2_visual.py:214-215` | Applies hole-filling, spatial, and temporal filters |
| 3. Rotation | `go2_visual.py:218` | Rotates 180° (`np.rot90(k=2)`) because D435i is mounted inverted |
| 4. Cropping | `go2_visual.py:222-224` | Crops top/bottom/left/right pixels |
| 5. Clip & Normalize | `go2_visual.py:226` | Clips to `[depth_min, depth_max]` then normalizes to `[0, 1]` |
| 6. Resize | `go2_visual.py:227` | Resizes to `output_resolution` via `resize2d()` (adaptive avg pool) |
| 7. Visual Encoder | `go2_visual.py:265` | Passes through `visual_encoder` neural network to get embedding |
| 8. Publish | `go2_visual.py:247-250` | Publishes embedding via ROS topic |

### Key Real-World Code Snippets

**Depth capture and filtering** (`go2_visual.py:188-227`):
```python
def get_depth_frame(self):
    # Wait for frames from realsense
    rs_frame = self.rs_pipeline.wait_for_frames(...)
    depth_frame = rs_frame.get_depth_frame()
    
    # Apply realsense filters (hole filling, spatial, temporal)
    for rs_filter in self.rs_filters:
        depth_frame = rs_filter.process(depth_frame)
    
    # Convert to numpy and rotate 180° (camera mounted inverted)
    depth_image_np = np.asanyarray(depth_frame.get_data())
    depth_image_np = np.rot90(depth_image_np, k=2)
    
    # Convert to torch tensor and crop
    depth_image_pyt = torch.from_numpy(depth_image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    depth_image_pyt = depth_image_pyt[:, :, crop_top:-crop_bottom-1, crop_left:-crop_right-1]
    
    # Clip and normalize to [0, 1]
    depth_image_pyt = torch.clip(depth_image_pyt, self.depth_range[0], self.depth_range[1]) \
                      / (self.depth_range[1] - self.depth_range[0])
    
    # Resize to network input resolution
    depth_image_pyt = resize2d(depth_image_pyt, self.output_resolution)
    
    return depth_image_pyt
```

**Realsense filter configuration** (`go2_visual.py:90-105`):
```python
self.rs_hole_filling_filter = rs.hole_filling_filter()
self.rs_spatial_filter = rs.spatial_filter()
self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
self.rs_temporal_filter = rs.temporal_filter()
self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
self.rs_filters = [
    self.rs_hole_filling_filter,
    self.rs_spatial_filter,
    self.rs_temporal_filter,
]
```

---

## 2. Simulation Depth Processing

### Primary Files

| File | Purpose |
|------|---------|
| `legged_gym/legged_gym/envs/base/legged_robot.py` | Base sensor setup and camera creation |
| `legged_gym/legged_gym/envs/base/legged_robot_noisy.py` | Depth image processing with noise simulation |

### Simulation Pipeline

| Step | Code Location | Function |
|------|---------------|----------|
| 1. Camera Setup | `legged_robot.py:1261-1274` | `_create_onboard_camera()` - Creates IsaacGym camera sensor |
| 2. Depth Capture | `legged_robot.py:1178-1185` | Wraps GPU tensor for `gymapi.IMAGE_DEPTH` |
| 3. Depth Sign Fix | `legged_robot_noisy.py:602` | Multiplies by `-1` (IsaacGym depth is negative) |
| 4. Noise Simulation | `legged_robot_noisy.py:603-615` | Adds stereo camera artifacts (see below) |
| 5. Normalize | `legged_robot_noisy.py:626` | `_normalize_depth_images()` - Clips & normalizes to `[0, 1]` |
| 6. Crop | `legged_robot_noisy.py:627` | `_crop_depth_images()` - Crops to configured region |
| 7. Resize | `legged_robot_noisy.py:628-629` | Optional bicubic resize to `output_resolution` |
| 8. Latency Buffer | `legged_robot_noisy.py:731-759` | `_get_forward_depth_obs()` - Simulates sensor latency |

### Main Processing Function (`legged_robot_noisy.py:598-631`):
```python
@torch.no_grad()
def _process_depth_image(self, depth_images):
    # depth_images: list of N tensors with shape (H, W)
    # Reverse the negative depth (IsaacGym convention)
    depth_images_ = torch.stack(depth_images).unsqueeze(1).contiguous().detach().clone() * -1
    
    # Apply noise simulation to bridge sim-to-real gap
    if hasattr(self.cfg.noise, "forward_depth"):
        if getattr(self.cfg.noise.forward_depth, "countour_threshold", 0.) > 0.:
            depth_images_ = self._add_depth_contour(depth_images_)
        if getattr(self.cfg.noise.forward_depth, "artifacts_prob", 0.) > 0.:
            depth_images_ = self._add_depth_artifacts(depth_images_, ...)
        if getattr(self.cfg.noise.forward_depth, "stereo_min_distance", 0.) > 0.:
            depth_images_ = self._add_depth_stereo(depth_images_)
        if getattr(self.cfg.noise.forward_depth, "sky_artifacts_prob", 0.) > 0.:
            depth_images_ = self._add_sky_artifacts(depth_images_)
    
    # Normalize to [0, 1]
    depth_images_ = self._normalize_depth_images(depth_images_)
    
    # Crop to desired region
    depth_images_ = self._crop_depth_images(depth_images_)
    
    # Optional resize
    if hasattr(self, "forward_depth_resize_transform"):
        depth_images_ = self.forward_depth_resize_transform(depth_images_)
    
    depth_images_ = depth_images_.clip(0, 1)
    return depth_images_.unsqueeze(0)  # (1, N, 1, H, W)
```

---

## 3. Noise Simulation Functions (Sim-to-Real Domain Gap)

These functions simulate the artifacts and noise patterns typical of stereo depth cameras (Intel RealSense D435i).

### Noise Simulation Overview

| Function | Code Location | Purpose |
|----------|---------------|---------|
| `_add_depth_contour()` | Line 380-388 | Simulates edge detection artifacts at depth discontinuities |
| `_add_depth_artifacts()` | Line 429-484 | Simulates random patch artifacts |
| `_add_depth_stereo()` | Line 494-543 | Simulates stereo camera limitations (near/far noise, block artifacts) |
| `_add_sky_artifacts()` | Line 553-577 | Simulates sky/ceiling artifacts when pointing upward |

### `_add_depth_contour()` - Edge Artifacts
```python
def _add_depth_contour(self, depth_images):
    # Uses 8 directional edge detection kernels
    # Max pools the edge responses and masks out high-gradient regions
    mask = F.max_pool2d(
        torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding=1)).max(dim=-3, keepdim=True)[0],
        kernel_size=self.cfg.noise.forward_depth.contour_detection_kernel_size,
        stride=1,
        padding=int(self.cfg.noise.forward_depth.contour_detection_kernel_size / 2),
    ) > self.cfg.noise.forward_depth.contour_threshold
    depth_images[mask] = 0.
    return depth_images
```

### `_add_depth_stereo()` - Stereo Camera Limitations
```python
def _add_depth_stereo(self, depth_images):
    N, _, H, W = depth_images.shape
    
    # Identify pixel categories based on distance
    far_mask = depth_images > self.cfg.noise.forward_depth.stereo_far_distance
    too_close_mask = depth_images < self.cfg.noise.forward_depth.stereo_min_distance
    near_mask = (~far_mask) & (~too_close_mask)
    
    # Add Gaussian noise to far regions
    far_noise = torch_rand_float(0., stereo_far_noise_std, (N, H * W), device=self.device).view(N, 1, H, W)
    depth_images += far_noise * far_mask
    
    # Add Gaussian noise to near regions
    near_noise = torch_rand_float(0., stereo_near_noise_std, (N, H * W), device=self.device).view(N, 1, H, W)
    depth_images += near_noise * near_mask
    
    # Handle too-close regions with block artifacts
    # ... (creates full-block and half-block artifacts)
    
    return depth_images
```

### `_add_sky_artifacts()` - Sky/Ceiling Artifacts
```python
def _add_sky_artifacts(self, depth_images):
    # Detects regions pointing to sky (all pixels above are also far)
    possible_to_sky_mask = depth_images > self.cfg.noise.forward_depth.sky_artifacts_far_distance
    to_sky_mask = self._recognize_top_down_seeing_sky(possible_to_sky_mask)
    
    # Add patch artifacts to sky regions
    # ... (creates randomized block artifacts)
    
    return depth_images
```

---

## 4. Pipeline Comparison Diagram

```
+-----------------------------------------------------------------------------+
|                         REAL-WORLD PIPELINE                                  |
|  +----------+   +------------+   +---------+   +-----------+   +----------+ |
|  |Realsense | ->| RS Filters | ->| Rotate  | ->| Crop+Norm | ->|  Resize  | |
|  |  D435i   |   |(hole/spatial|   | 180 deg |   |[0,3000]mm|   | to (H,W) | |
|  +----------+   | /temporal)  |   +---------+   +-----------+   +----------+ |
|                 +------------+                                           v  |
|                                                              +---------------+
|                                                              |Visual Encoder |
|                                                              |  -> Embedding |
|                                                              +---------------+
+-----------------------------------------------------------------------------+

+-----------------------------------------------------------------------------+
|                         SIMULATION PIPELINE                                  |
|  +----------+   +---------+   +---------------+   +-----------+   +-------+ |
|  |IsaacGym  | ->| x(-1)   | ->| Stereo Noise  | ->| Normalize | ->| Crop  | |
|  | Camera   |   |(fix sign)|   |(contour/arti- |   |  [0, 1]   |   |       | |
|  |(IMAGE_   |   +---------+   |facts/stereo/  |   +-----------+   +---+---+ |
|  | DEPTH)   |                 |   sky)        |                     v       |
|  +----------+                 +---------------+               +---------+  |
|                                                                | Resize  |  |
|                                                                +----+----+  |
|                                                                     v       |
|                                                              +-------------+
|                                                              |Latency Buffer|
|                                                              | (sim delay)  |
|                                                              +-------------+
+-----------------------------------------------------------------------------+
```

---

## 5. Configuration Parameters

### Go2 Config (`legged_gym/envs/go2/go2_distill_config.py:112-127`):
```python
class forward_depth:
    stereo_min_distance = 0.175  # meters (for 480x640 resolution)
    stereo_far_distance = 1.2    # meters
    stereo_far_noise_std = 0.08
    stereo_near_noise_std = 0.02
    stereo_full_block_artifacts_prob = 0.008
    stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
    stereo_full_block_height_mean_std = [62, 1.5]
    stereo_full_block_width_mean_std = [3, 0.01]
    stereo_half_block_spark_prob = 0.02
    stereo_half_block_value = 3000
    sky_artifacts_prob = 0.0001
    sky_artifacts_far_distance = 2.0
    sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
    sky_artifacts_height_mean_std = [2, 3.2]
    sky_artifacts_width_mean_std = [2, 3.2]
```

### Go1/A1 Config (`legged_gym/envs/go1/go1_field_distill_config.py:205-220`):
```python
class forward_depth:
    stereo_min_distance = 0.12  # meters (for 240x424 resolution)
    stereo_far_distance = 2.0   # meters
    stereo_far_noise_std = 0.08
    stereo_near_noise_std = 0.02
    stereo_full_block_artifacts_prob = 0.004
    stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
    stereo_full_block_height_mean_std = [62, 1.5]
    stereo_full_block_width_mean_std = [3, 0.01]
    stereo_half_block_spark_prob = 0.02
    stereo_half_block_value = 3000
    sky_artifacts_prob = 0.0001
    sky_artifacts_far_distance = 2.0
    sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
    sky_artifacts_height_mean_std = [2, 3.2]
    sky_artifacts_width_mean_std = [2, 3.2]
```

---

## 6. Key Differences Summary

| Aspect | Real-World | Simulation |
|--------|------------|------------|
| **Depth Source** | Intel RealSense D435i | IsaacGym `IMAGE_DEPTH` |
| **Depth Sign** | Positive values | Negative (requires `x(-1)`) |
| **Filtering** | Realsense SDK filters | Custom noise simulation |
| **Rotation** | 180° (inverted mount) | None |
| **Noise Model** | Hardware noise | Simulated stereo artifacts |
| **Latency** | Hardware latency | Configurable buffer simulation |

---

## 7. Common Normalization Formula

Both pipelines normalize depth to `[0, 1]` range:

```python
# Convert meters to millimeters for real-world
depth_mm = depth_m * 1000

# Clip to valid range
depth_clipped = clip(depth_mm, depth_min, depth_max)

# Normalize to [0, 1]
depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min)
```

Default depth range: `[0.0, 3.0]` meters = `[0, 3000]` mm
