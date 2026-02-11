"""
gsplat_viewer.py - Interactive Gaussian Splatting Viewer

Controls:
  WASD - Move camera (forward/left/back/right)
  QE - Move up/down
  Arrow keys - Rotate camera (pitch/yaw)
  +/- - Zoom in/out (adjust FOV)
  R - Reset camera
  ESC - Quit
"""

import os
import sys
import numpy as np
import torch
import pygame
from pygame.locals import *
from gsplat import rasterization


def load_ply(filepath):
    """Load standard 3DGS PLY file."""
    import struct

    with open(filepath, 'rb') as f:
        # Parse header
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break

        header_str = header.decode('ascii')

        # Get vertex count
        for line in header_str.split('\n'):
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
                break

        print(f"Loading {n_vertices} Gaussians...")

        # Read binary data (17 floats per vertex)
        # x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        data = np.frombuffer(f.read(n_vertices * 17 * 4), dtype=np.float32)
        data = data.reshape(n_vertices, 17)

        xyz = data[:, 0:3]
        # normals = data[:, 3:6]  # Not used
        sh = data[:, 6:9]  # f_dc (SH degree 0)
        opacity = data[:, 9:10]  # logit
        scale = data[:, 10:13]  # log scale
        rot = data[:, 13:17]  # quaternion (w, x, y, z)

        return {
            'xyz': xyz,
            'sh': sh,
            'opacity': opacity,
            'scale': scale,
            'rot': rot
        }


def create_camera_matrix(position, yaw, pitch, target=None):
    """Create view matrix using look-at style construction."""
    if target is None:
        # Compute forward direction from yaw/pitch
        forward = np.array([
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch)
        ])
        target = position + forward

    # Look-at matrix construction
    forward = target - position
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # View matrix (world to camera)
    # Camera looks along -Z in its local frame
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[0, :3] = right
    viewmat[1, :3] = up
    viewmat[2, :3] = -forward  # Camera looks along -Z
    viewmat[:3, 3] = -viewmat[:3, :3] @ position

    return viewmat


def main():
    # Configuration
    ply_path = sys.argv[1] if len(sys.argv) > 1 else "output_gaussians.ply"

    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found: {ply_path}")
        print("Usage: python gsplat_viewer.py [path_to_ply]")
        sys.exit(1)

    # Load Gaussians
    data = load_ply(ply_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to tensors - ensure float32
    means = torch.from_numpy(data['xyz'].copy()).float().to(device)

    # Quaternions: PLY has (w, x, y, z), gsplat expects (w, x, y, z)
    quats_raw = torch.from_numpy(data['rot'].copy()).float().to(device)
    # Try both quaternion orders - gsplat might expect (x, y, z, w)
    # Uncomment next line if scene looks wrong:
    # quats_raw = quats_raw[:, [1, 2, 3, 0]]  # Convert wxyz -> xyzw
    quats = quats_raw / quats_raw.norm(dim=-1, keepdim=True)  # Normalize

    # Scales: Check if PLY has log scale or actual scale
    scales_raw = torch.from_numpy(data['scale'].copy()).float().to(device)
    print(f"  Raw scale range: [{scales_raw.min().item():.4f}, {scales_raw.max().item():.4f}]")

    # If raw scales are small positive numbers, they might already be actual scales
    # If raw scales are around 0 (positive/negative), they're log scales
    if scales_raw.min() > 0 and scales_raw.max() < 10:
        # Already actual scales - don't apply exp
        print("  -> Treating as actual scales (no exp)")
        scales = scales_raw
    else:
        # Log scales - apply exp
        print("  -> Treating as log scales (applying exp)")
        scales = torch.exp(scales_raw)

    # Opacities: Check if PLY has logit or actual opacity
    opacity_raw = torch.from_numpy(data['opacity'].copy()).float().to(device).squeeze(-1)
    print(f"  Raw opacity range: [{opacity_raw.min().item():.4f}, {opacity_raw.max().item():.4f}]")

    # If raw opacities are in [0,1], they're already actual opacities
    # If raw opacities are outside [0,1], they're logits
    if opacity_raw.min() >= 0 and opacity_raw.max() <= 1:
        print("  -> Treating as actual opacities (no sigmoid)")
        opacities = opacity_raw
    else:
        print("  -> Treating as logit opacities (applying sigmoid)")
        opacities = torch.sigmoid(opacity_raw)

    # Colors: SH degree 0 coefficients
    # Convert from SH to RGB: color = SH_C0 * sh + 0.5
    SH_C0 = 0.28209479177387814
    sh_raw = torch.from_numpy(data['sh'].copy()).float().to(device)
    colors = torch.clamp(SH_C0 * sh_raw + 0.5, 0, 1)

    print(f"Loaded: {means.shape[0]} Gaussians")
    print(f"  Position range: [{means.min().item():.2f}, {means.max().item():.2f}]")
    print(f"  Scale range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")
    print(f"  Opacity range: [{opacities.min().item():.4f}, {opacities.max().item():.4f}]")
    print(f"  Color range: [{colors.min().item():.4f}, {colors.max().item():.4f}]")

    # Debug: check for NaN/Inf
    if torch.isnan(means).any() or torch.isinf(means).any():
        print("WARNING: NaN/Inf in means!")
    if torch.isnan(scales).any() or torch.isinf(scales).any():
        print("WARNING: NaN/Inf in scales!")
    if torch.isnan(quats).any() or torch.isinf(quats).any():
        print("WARNING: NaN/Inf in quats!")
    if torch.isnan(opacities).any() or torch.isinf(opacities).any():
        print("WARNING: NaN/Inf in opacities!")
    if torch.isnan(colors).any() or torch.isinf(colors).any():
        print("WARNING: NaN/Inf in colors!")

    # Check data types
    print(f"  Data types: means={means.dtype}, quats={quats.dtype}, scales={scales.dtype}, opacities={opacities.dtype}, colors={colors.dtype}")

    # Initialize pygame
    pygame.init()
    width, height = 1280, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"gsplat Viewer - {os.path.basename(ply_path)}")
    clock = pygame.time.Clock()

    # Camera state
    scene_center = means.mean(dim=0)
    scene_extent = (means.max(dim=0).values - means.min(dim=0).values).max().item()

    print(f"  Scene center: {scene_center.cpu().numpy()}")
    print(f"  Scene extent: {scene_extent:.2f}")

    # CENTER THE MEANS AT ORIGIN - this is key for gsplat rendering
    means = means - scene_center
    print(f"  Centered means range: [{means.min().item():.2f}, {means.max().item():.2f}]")

    # Camera position relative to origin (where scene center now is)
    cam_pos = np.array([0.0, 0.0, scene_extent * 1.5], dtype=np.float32)
    cam_yaw = 0.0
    cam_pitch = 0.0
    fov = 60.0

    move_speed = scene_extent * 0.05
    rot_speed = 0.03

    # Camera intrinsics
    focal = width / (2 * np.tan(np.radians(fov) / 2))

    # Scale multiplier - scales are tiny (0.001-0.3) for a 32-unit scene
    # Need large multiplier to make Gaussians visible
    scale_mult = 100.0

    print("\nControls:")
    print("  WASD - Move camera")
    print("  QE - Up/Down")
    print("  Arrows - Rotate")
    print("  +/- - Zoom (FOV)")
    print("  [/] - Decrease/Increase Gaussian scale")
    print("  R - Reset")
    print("  P - Print camera position")
    print("  T - Test render single Gaussian")
    print("  G - Test render scene Gaussians with simple camera")
    print("  ESC - Quit")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    # Reset camera (scene is centered at origin)
                    cam_pos = np.array([0.0, 0.0, scene_extent * 1.5], dtype=np.float32)
                    cam_yaw = 0.0
                    cam_pitch = 0.0
                    fov = 60.0
                    scale_mult = 100.0
                elif event.key == K_p:
                    print(f"Camera: pos={cam_pos}, yaw={cam_yaw:.2f}, pitch={cam_pitch:.2f}")
                elif event.key == K_LEFTBRACKET:
                    scale_mult *= 0.8
                    print(f"Scale multiplier: {scale_mult:.4f}")
                elif event.key == K_RIGHTBRACKET:
                    scale_mult *= 1.25
                    print(f"Scale multiplier: {scale_mult:.4f}")
                elif event.key == K_g:
                    # Test with simple camera (means are already centered at origin)
                    print("Testing with scene Gaussians...")
                    simple_viewmat = torch.eye(4, device=device, dtype=torch.float32)
                    simple_viewmat[2, 3] = scene_extent  # Camera at z=scene_extent looking at origin

                    n_test = min(10000, means.shape[0])
                    test_renders, test_alphas, _ = rasterization(
                        means=means[:n_test],
                        quats=quats[:n_test],
                        scales=scales[:n_test] * scale_mult,
                        opacities=opacities[:n_test],
                        colors=colors[:n_test],
                        viewmats=simple_viewmat[None],
                        Ks=K,
                        width=width, height=height,
                        near_plane=0.01, far_plane=1000.0,
                        packed=False,
                    )
                    print(f"  Render max: {test_renders.max().item():.4f}, alpha max: {test_alphas.max().item():.4f}")
                    print(f"  Means range: [{means[:n_test].min().item():.2f}, {means[:n_test].max().item():.2f}]")
                elif event.key == K_t:
                    # Test: render a single large Gaussian at scene center
                    print("Testing with single Gaussian at scene center...")
                    print(f"  Camera at: {cam_pos}, looking at: {scene_center}")

                    # Simple test: Gaussian at origin, camera looking at it
                    test_means = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
                    test_quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
                    test_scales = torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=torch.float32)
                    test_opacities = torch.tensor([1.0], device=device, dtype=torch.float32)
                    test_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)  # Red

                    # Simple view matrix: camera at z=5 looking at origin
                    test_viewmat = torch.tensor([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 5.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ], device=device, dtype=torch.float32)[None]

                    test_K = torch.tensor([
                        [focal, 0, width/2],
                        [0, focal, height/2],
                        [0, 0, 1]
                    ], device=device, dtype=torch.float32)[None]

                    print(f"  viewmat:\n{test_viewmat[0]}")
                    print(f"  K:\n{test_K[0]}")

                    test_renders, test_alphas, meta = rasterization(
                        means=test_means, quats=test_quats, scales=test_scales,
                        opacities=test_opacities, colors=test_colors,
                        viewmats=test_viewmat, Ks=test_K, width=width, height=height,
                        near_plane=0.01, far_plane=100.0,
                        packed=False,
                    )
                    print(f"  Test render shape: {test_renders.shape}")
                    print(f"  Test render max: {test_renders.max().item():.4f}, min: {test_renders.min().item():.4f}")
                    print(f"  Test alpha max: {test_alphas.max().item():.4f}")
                    if 'radii' in meta:
                        print(f"  Radii: {meta['radii']}")

        # Continuous key presses
        keys = pygame.key.get_pressed()

        # Calculate forward and right vectors
        forward = np.array([
            np.sin(cam_yaw) * np.cos(cam_pitch),
            -np.sin(cam_pitch),
            -np.cos(cam_yaw) * np.cos(cam_pitch)
        ])
        right = np.array([np.cos(cam_yaw), 0, np.sin(cam_yaw)])
        up = np.array([0, 1, 0])

        # Movement
        if keys[K_w]:
            cam_pos += forward * move_speed
        if keys[K_s]:
            cam_pos -= forward * move_speed
        if keys[K_a]:
            cam_pos -= right * move_speed
        if keys[K_d]:
            cam_pos += right * move_speed
        if keys[K_q]:
            cam_pos += up * move_speed
        if keys[K_e]:
            cam_pos -= up * move_speed

        # Rotation
        if keys[K_LEFT]:
            cam_yaw -= rot_speed
        if keys[K_RIGHT]:
            cam_yaw += rot_speed
        if keys[K_UP]:
            cam_pitch = max(-np.pi/2 + 0.1, cam_pitch - rot_speed)
        if keys[K_DOWN]:
            cam_pitch = min(np.pi/2 - 0.1, cam_pitch + rot_speed)

        # Zoom
        if keys[K_EQUALS] or keys[K_PLUS]:
            fov = max(20, fov - 1)
        if keys[K_MINUS]:
            fov = min(120, fov + 1)

        # Update focal length
        focal = width / (2 * np.tan(np.radians(fov) / 2))

        # Build rotation matrix from yaw/pitch
        # Yaw: rotation around Y axis, Pitch: rotation around X axis
        cy, sy = np.cos(cam_yaw), np.sin(cam_yaw)
        cp, sp = np.cos(cam_pitch), np.sin(cam_pitch)

        # Combined rotation: R = Ry(yaw) @ Rx(pitch)
        R = np.array([
            [cy, sy*sp, sy*cp],
            [0, cp, -sp],
            [-sy, cy*sp, cy*cp]
        ], dtype=np.float32)

        # gsplat viewmat: rotation + camera position (NOT -R @ cam_pos!)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = R @ cam_pos  # Positive, rotated camera position
        viewmat_tensor = torch.from_numpy(viewmat).to(device)[None]

        # Camera intrinsics
        K = torch.tensor([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1],
        ], device=device, dtype=torch.float32)[None]

        # Render
        with torch.no_grad():
            renders, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales * scale_mult,  # Apply scale multiplier
                opacities=opacities,
                colors=colors,
                viewmats=viewmat_tensor,
                Ks=K,
                width=width,
                height=height,
                packed=False,  # Try non-packed mode
                near_plane=0.01,
                far_plane=1000.0,
            )

        # Debug: print render stats occasionally
        if pygame.time.get_ticks() % 2000 < 20:  # Every ~2 seconds
            print(f"Render max: {renders.max().item():.4f}, alpha max: {alphas.max().item():.4f}")

        # Convert to numpy for pygame
        image = renders[0].cpu().numpy()
        alpha = alphas[0].cpu().numpy()

        # Composite with gray background to see alpha
        bg_color = np.array([0.2, 0.2, 0.3])
        image = image * alpha + bg_color * (1 - alpha)

        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # Display
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

        # Show FPS and info
        fps = clock.get_fps()
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {fps:.1f} | Scale: {scale_mult:.3f} | FOV: {fov:.0f}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
