import einops
import torch
# harim
import torch.nn.functional as F

# --- Intrinsics Transformations ---

def normalize_intrinsics(intrinsics, image_shape):
    '''Normalize an intrinsics matrix given the image shape'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= image_shape[1]
    intrinsics[..., 1, :] /= image_shape[0]
    return intrinsics


def unnormalize_intrinsics(intrinsics, image_shape):
    '''Unnormalize an intrinsics matrix given the image shape'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] *= image_shape[1]
    intrinsics[..., 1, :] *= image_shape[0]
    return intrinsics


# --- Quaternions, Rotations and Scales ---

def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    '''
    Convert the 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    '''

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(scale, rotation_xyzw):
    '''Build the 3x3 covariance matrix from the three dimensional scale and the
    four dimension quaternion'''
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ einops.rearrange(scale, "... i j -> ... j i")
        @ einops.rearrange(rotation, "... i j -> ... j i")
    )


# --- Projections ---

def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points


# harim

def constrain_to_ray(pts, K):
    """
    pts: (B, H, W, 3) - MAST3R가 예측한 3D 포인트맵
    K: (B, 3, 3) - 카메라 내동작 파라미터
    """
    B, H, W, _ = pts.shape
    device = pts.device

    # 1. 이미지 평면의 픽셀 그리드 생성 (u, v)
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), indexing='ij')
    pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (H, W, 3)
    pixel_coords = pixel_coords.view(1, H*W, 3).expand(B, -1, -1) # (B, HW, 3)
    
    # 2. K의 역행렬을 이용해 각 픽셀의 방향 벡터(Ray) 계산
    # K_inv * [u, v, 1]^T
    K_inv = torch.inverse(K)
    rays = torch.bmm(K_inv, pixel_coords.transpose(1, 2)).transpose(1, 2) # (B, H*W, 3)
    rays = rays.view(B, H, W, 3)

    # 3. 모델이 예측한 점의 깊이(Z) 추출
    # P_new = (Ray_direction / Ray_z) * Predicted_Z
    depth = pts[..., 2:3] 
    corrected_pts = (rays / (rays[..., 2:3] + 1e-8)) * depth

    return corrected_pts

def solve_sim3_alignment(pts_a, pts_b):
    """
    pts_a (Source): (N, 3)
    pts_b (Target): (N, 3)
    Target = s * R * Source + t 공식을 만족하는 파라미터 산출
    """
    # 1. 각 점 집합의 중심(Centroid) 계산
    mu_a = pts_a.mean(dim=0)
    mu_b = pts_b.mean(dim=0)

    # 2. 중심 이동 (Centering)
    a_centered = pts_a - mu_a
    b_centered = pts_b - mu_b

    # 3. 스케일 계산을 위한 분산(Variance) 산출
    var_a = torch.mean(torch.sum(a_centered**2, dim=1))

    # 4. SVD를 이용한 회전 행렬(R) 계산
    # Covariance H = a_centered^T * b_centered
    H = torch.mm(a_centered.t(), b_centered)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.t()
    
    # 반전(Reflection) 방지 로직
    d = torch.det(torch.mm(V, U.t()))
    S_matrix = torch.eye(3, device=pts_a.device)
    if d < 0:
        S_matrix[2, 2] = -1

    R = torch.mm(V, torch.mm(S_matrix, U.t()))

    # 5. 스케일(s) 계산
    # s = trace(S * S_matrix) / var_a
    # 여기서는 좀 더 안정적인 비율 방식을 사용 가능
    s = torch.trace(torch.mm(torch.diag(S), S_matrix)) / var_a

    # 6. 이동 벡터(t) 계산
    t = mu_b - s * torch.mv(R, mu_a)

    return s, R, t

def transform_points(points, T_WC):
    """
    points: (B, H, W, 3) 또는 (N, 3)
    T_WC: (B, 4, 4) 또는 (4, 4) 포즈 행렬
    """
    # points를 Homogeneous 좌표로 변환 후 T_WC 곱하기
    R = T_WC[..., :3, :3]
    t = T_WC[..., :3, 3:4].transpose(-1, -2)
    return (points @ R.transpose(-1, -2)) + t

def transform_rotations(rotations, T_WC):
    """
    rotations: (B, N, 4) 쿼터니언 형태 (또는 모델의 로테이션 표현)
    T_WC: (B, 4, 4)
    """
    # 단순 가우시안 로테이션은 T_WC의 R 성분을 곱하여 전역 방향으로 회전시킵니다.
    # (쿼터니언 연산 또는 행렬 연산 필요)
    pass


def refine_pose_gn(pts_src, pts_tgt, initial_guess, confidence=None, iterations=3):
    """
    Gauss-Newton optimization for Sim(3) alignment.
    """
    s, R, t = initial_guess
    device = pts_src.device
    N = pts_src.shape[0]

    # Confidence가 없을 경우 모든 점에 동일한 가중치(1) 부여
    if confidence is None:
        confidence = torch.ones((N, 1), device=device)
    else:
        confidence = confidence.view(N, 1)

    # 픽셀별 가중치를 3D 좌표(x,y,z)에 대응하도록 확장
    W = confidence.repeat_interleave(3).reshape(-1, 1)

    for _ in range(iterations):
        # 1. 현재 파라미터로 변환
        pts_rot = pts_src @ R.t()
        pts_pred = s * pts_rot + t
        
        # 2. 잔차 계산
        residuals = (pts_tgt - pts_pred).reshape(-1, 1)

        # 3. 자코비안 구성
        J = torch.zeros((N * 3, 7), device=device)
        J[:, :3] = torch.eye(3, device=device).repeat(N, 1) # Translation
        
        p = s * pts_rot
        skew_p = torch.zeros((N, 3, 3), device=device)
        skew_p[:, 0, 1], skew_p[:, 0, 2] = -p[:, 2],  p[:, 1]
        skew_p[:, 1, 0], skew_p[:, 1, 2] =  p[:, 2], -p[:, 0]
        skew_p[:, 2, 0], skew_p[:, 2, 1] = -p[:, 1],  p[:, 0]
        J[:, 3:6] = skew_p.reshape(N * 3, 3) # Rotation
        
        J[:, 6:7] = pts_rot.reshape(N * 3, 1) # Scale

        # 4. 가중치 적용 Normal Equation (H * delta = g)
        weighted_J = J * W
        H = J.t() @ weighted_J
        g = J.t() @ (residuals * W)

        H += 1e-6 * torch.eye(7, device=device) # Damping

        # 5. 해 구하기
        try:
            delta = torch.linalg.solve(H, g).squeeze()
        except RuntimeError: # Singular matrix 방어
            break

        # 6. 업데이트
        t = t + delta[:3]
        s = s + delta[6]
        
        dr = delta[3:6]
        dR_skew = torch.tensor([
            [0, -dr[2], dr[1]],
            [dr[2], 0, -dr[0]],
            [-dr[1], dr[0], 0]
        ], device=device)
        R = (torch.eye(3, device=device) + dR_skew) @ R
        
        # 정규화
        U, _, Vh = torch.linalg.svd(R)
        R = U @ Vh

    return s, R, t