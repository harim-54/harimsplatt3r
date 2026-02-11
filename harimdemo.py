"""
Multi-view Gaussian Splatting with MAST3R Global Alignment

This script properly handles multiple images by:
1. Using MAST3R's sparse_global_alignment for globally consistent camera poses
2. Generating Gaussians for each image pair using Splatt3r
3. Transforming Gaussians to world coordinates using the global poses
4. Merging all Gaussians into a single PLY file

Author: Based on MAST3R/Splatt3r pipeline
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Path setup for MAST3R/DUST3R/Splatt3r
base_path = os.path.dirname(os.path.abspath(__file__))
paths_to_add = [
    base_path,
    os.path.join(base_path, 'src/mast3r_src'),
    os.path.join(base_path, 'src/mast3r_src/dust3r'),
]
for p in paths_to_add:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# HEIC support
from pillow_heif import register_heif_opener
register_heif_opener()

# MAST3R imports
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.model import AsymmetricMASt3R
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

# Splatt3r imports
import main as splatt3r_main
from huggingface_hub import hf_hub_download
from utils.geometry import camera_space_to_world_space


class GlobalAlignedGaussianEngine:
    """
    Engine that combines MAST3R's global alignment with Splatt3r's Gaussian prediction.
    """

    def __init__(self, splatt3r_model, mast3r_model, device):
        self.splatt3r_model = splatt3r_model
        self.mast3r_model = mast3r_model
        self.device = device
        self.all_gaussians = []

    def run_global_alignment(self, image_files, cache_path='./cache_global'):
        """
        Run MAST3R's sparse global alignment on all images.
        Returns globally consistent camera poses.
        """
        print("\n" + "="*60)
        print("Step 1: Loading images and running global alignment")
        print("="*60)

        # Load images using dust3r's loader (handles HEIC, resizing, etc.)
        imgs = load_images(image_files, size=512, square_ok=True, verbose=True)

        if len(imgs) < 2:
            raise ValueError("Need at least 2 images for reconstruction")

        # Create pairs - 'complete' means all-to-all pairs for best accuracy
        # For many images, consider 'swin-3' (sliding window) for efficiency
        n_imgs = len(imgs)
        if n_imgs <= 5:
            scene_graph = 'complete'
        else:
            scene_graph = 'swin-3'  # Sliding window with size 3

        print(f"\nCreating image pairs with scene graph: {scene_graph}")
        pairs = make_pairs(imgs, scene_graph=scene_graph, symmetrize=True)
        print(f"Created {len(pairs)} pairs from {n_imgs} images")

        # Run sparse global alignment
        print("\nRunning sparse global alignment (this may take a while)...")
        os.makedirs(cache_path, exist_ok=True)

        scene = sparse_global_alignment(
            image_files,
            pairs,
            cache_path,
            self.mast3r_model,
            device=self.device,
            lr1=0.07,      # Coarse learning rate
            niter1=500,    # Coarse iterations
            lr2=0.014,     # Fine learning rate
            niter2=200,    # Fine iterations
            shared_intrinsics=False,
        )

        print("\nGlobal alignment complete!")
        return scene, imgs

    def generate_gaussians_for_pairs(self, imgs, scene):
        """
        Generate Gaussians for consecutive image pairs using Splatt3r,
        then transform them to world coordinates using global poses.
        """
        print("\n" + "="*60)
        print("Step 2: Generating Gaussians for each pair")
        print("="*60)

        # Get globally aligned camera poses (camera-to-world transforms)
        cam2world = scene.get_im_poses()  # Shape: [N, 4, 4]

        print(f"\nGlobal camera poses shape: {cam2world.shape}")

        # Process consecutive pairs
        for i in range(len(imgs) - 1):
            view1 = self._prepare_view(imgs[i])
            view2 = self._prepare_view(imgs[i + 1])

            print(f"\nProcessing pair ({i}, {i+1})...")

            with torch.no_grad():
                pred1, pred2 = self.splatt3r_model(view1, view2)

                # [하림 : Gauss-Newton 정밀 최적화 적용 시작]
                # 1. 대응점 추출 (view1 기준)
                pts1 = pred1['pts3d'].reshape(-1, 3)
                # pred2가 view1에서 바라본 위치(means_in_other_view)를 가져옴
                pts2 = pred2['means_in_other_view'].reshape(-1, 3)

                # 2. 초기 Sim3 최적화 (SVD 기반)
                # geometry.solve_sim3_alignment 함수가 구현되어 있어야 합니다.
                s, R, t = geometry.solve_sim3_alignment(pts1, pts2)
                
                # 3. 하림님의 GN 정밀 최적화 수행
                # main.py에서 우리가 추가한 refine_pose_gn 메서드를 호출합니다.
                refined_s, refined_R, refined_t = self.splatt3r_model.refine_pose_gn(pts1, pts2, (s, R, t))

                # 4. 정밀하게 최적화된 로컬 중심점 계산 (view1의 좌표계 내에서 정렬)
                # local_means_refined는 이제 pts1(view1)과 완벽하게 정렬된 상태입니다.
                local_means_refined = refined_s * (pred2['means_in_other_view'] @ refined_R.t()) + refined_t

            # 5. 이제 전역(World) 좌표계로 변환
            # MAST3R의 Global Pose (c2w_view1)를 사용하여 월드로 보냅니다.
            c2w_view1 = cam2world[i]  # [4, 4]
            
            # 정렬된 로컬 좌표를 월드 좌표로 변환: P_world = R_c2w * P_local + t_c2w
            world_means = (local_means_refined.reshape(-1, 3) @ c2w_view1[:3, :3].t()) + c2w_view1[:3, 3]
            # [하림 로직 적용 끝]

            # Extract Gaussian parameters (keep raw values for viewer compatibility)
            sh = pred2['sh'].reshape(-1, 3)
            scales = pred2['scales'].reshape(-1, 3)  # Keep log scale
            rotations = pred2['rotations'].reshape(-1, 4)
            opacities = pred2['opacities'].reshape(-1, 1)  # Keep logit

            # Transform rotations to world frame
            # The rotation quaternion needs to be composed with the camera rotation
            world_rotations = self._transform_rotations_to_world(rotations, c2w_view1[:3, :3])

            # Store Gaussian data
            self.all_gaussians.append({
                'xyz': world_means.cpu().numpy().astype(np.float32),
                'sh': sh.cpu().numpy().astype(np.float32),
                'opacity': opacities.cpu().numpy().astype(np.float32),
                'scale': scales.cpu().numpy().astype(np.float32),
                'rot': world_rotations.cpu().numpy().astype(np.float32),
            })

            print(f"  -> Added {world_means.shape[0]} Gaussians")

        # Also process the first frame (which is skipped in pairwise processing)
        # Use the first pair but take pred1's Gaussians
        if len(imgs) >= 2:
            view1 = self._prepare_view(imgs[0])
            view2 = self._prepare_view(imgs[1])

            print(f"\nProcessing first frame (0) separately...")

            with torch.no_grad():
                pred1, pred2 = self.splatt3r_model(view1, view2)

            c2w_view1 = cam2world[0]

            # pred1's points are already in view1's frame
            local_means = pred1['means'] if 'means' in pred1 else pred1['pts3d']

            # Reshape for transformation: need [b, v, h, w, 3] format
            if local_means.dim() == 4:
                local_means_5d = local_means.unsqueeze(1)  # [1, 1, H, W, 3]
            else:
                local_means_5d = local_means.view(1, 1, -1, 1, 3)

            c2w_4d = c2w_view1.view(1, 1, 4, 4)
            world_means = camera_space_to_world_space(local_means_5d, c2w_4d)
            world_means = world_means.reshape(-1, 3)

            sh = pred1['sh'].reshape(-1, 3)
            scales = pred1['scales'].reshape(-1, 3)  # Keep log scale
            rotations = pred1['rotations'].reshape(-1, 4)
            opacities = pred1['opacities'].reshape(-1, 1)  # Keep logit

            world_rotations = self._transform_rotations_to_world(rotations, c2w_view1[:3, :3])

            self.all_gaussians.insert(0, {
                'xyz': world_means.cpu().numpy().astype(np.float32),
                'sh': sh.cpu().numpy().astype(np.float32),
                'opacity': opacities.cpu().numpy().astype(np.float32),
                'scale': scales.cpu().numpy().astype(np.float32),
                'rot': world_rotations.cpu().numpy().astype(np.float32),
            })

            print(f"  -> Added {world_means.shape[0]} Gaussians from first frame")

    def _prepare_view(self, img_dict):
        """Prepare image dictionary for Splatt3r model."""
        true_shape = img_dict['true_shape']
        if isinstance(true_shape, np.ndarray):
            true_shape = torch.from_numpy(true_shape).to(self.device)
        elif isinstance(true_shape, torch.Tensor):
            true_shape = true_shape.to(self.device)
        return {
            'img': img_dict['img'].to(self.device),
            'original_img': img_dict['original_img'].to(self.device),
            'true_shape': true_shape,
            'instance': img_dict['instance'],
        }

    def _rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion (w,x,y,z) - pure PyTorch/CUDA"""
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return torch.stack([w, x, y, z])

    def _quaternion_multiply_batch(self, q1, q2):
        """Multiply quaternions: q1 * q2, batched - pure PyTorch/CUDA
        q1: [4] single quaternion (w,x,y,z)
        q2: [N, 4] batch of quaternions (w,x,y,z)
        Returns: [N, 4]
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=1)

    def _transform_rotations_to_world(self, quats, rotation_matrix):
        """
        Transform quaternions from local camera frame to world frame.
        All operations on GPU with CUDA.

        Args:
            quats: [N, 4] quaternions (w, x, y, z)
            rotation_matrix: [3, 3] camera rotation matrix (R in c2w)

        Returns:
            [N, 4] quaternions in world frame
        """
        cam_quat = self._rotation_matrix_to_quaternion(rotation_matrix)  # [4]
        return self._quaternion_multiply_batch(cam_quat, quats)  # [N, 4]

    def save_gaussians_ply(self, filename="global_aligned_gaussians.ply"):
        """Save all Gaussians to a SuperSplat-compatible PLY file."""
        if not self.all_gaussians:
            print("No Gaussians to save!")
            return

        print("\n" + "="*60)
        print("Step 3: Saving Gaussians to PLY")
        print("="*60)

        # Concatenate all Gaussians
        xyz = np.concatenate([g['xyz'] for g in self.all_gaussians], axis=0)
        sh = np.concatenate([g['sh'] for g in self.all_gaussians], axis=0)
        opacity = np.concatenate([g['opacity'] for g in self.all_gaussians], axis=0)
        scale = np.concatenate([g['scale'] for g in self.all_gaussians], axis=0)
        rot = np.concatenate([g['rot'] for g in self.all_gaussians], axis=0)

        # Normals (zeros)
        normals = np.zeros_like(xyz)

        print(f"Total Gaussians: {len(xyz)}")

        # PLY header for SuperSplat compatibility
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

        with open(filename, 'wb') as f:
            f.write(header.encode('ascii'))
            combined = np.hstack([xyz, normals, sh, opacity, scale, rot])
            f.write(combined.astype(np.float32).tobytes())

        print(f"\nSaved to: {filename}")
        print("Open with SuperSplat Editor: https://playcanvas.com/supersplat/editor")


def main():
    """Main entry point."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Load Models ===
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)

    # Load MAST3R model for global alignment
    print("Loading MAST3R model...")
    mast3r_model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    mast3r_model = mast3r_model.to(device).eval()

    # Load Splatt3r model for Gaussian prediction
    print("Loading Splatt3r model...")
    splatt3r_weights = hf_hub_download(
        repo_id="brandonsmart/splatt3r_v1.0",
        filename="epoch=19-step=1200.ckpt"
    )
    splatt3r_model = splatt3r_main.MAST3RGaussians.load_from_checkpoint(
        splatt3r_weights,
        map_location=device,
        weights_only=False  # PyTorch 2.6+ requires this for omegaconf objects
    ).eval()

    # === Prepare Image List ===
    # Scan input folder for images
    input_dir = os.path.join(base_path, "input")
    image_files = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif'))
    ])

    if not image_files:
        print(f"\nNo image files found in {input_dir}!")
        print("Supported formats: .jpg, .jpeg, .png, .heic, .heif")
        return

    existing_files = image_files

    print(f"\nFound {len(existing_files)} images:")
    for f in existing_files:
        print(f"  - {os.path.basename(f)}")

    # === Run Pipeline ===
    engine = GlobalAlignedGaussianEngine(splatt3r_model, mast3r_model, device)

    # Step 1: Global alignment
    cache_path = os.path.join(base_path, 'cache_global_alignment')
    scene, imgs = engine.run_global_alignment(existing_files, cache_path)

    # Step 2: Generate Gaussians with global poses
    engine.generate_gaussians_for_pairs(imgs, scene)

    # Step 3: Save PLY
    output_file = os.path.join(base_path, "global_aligned_gaussians.ply")
    engine.save_gaussians_ply(output_file)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
