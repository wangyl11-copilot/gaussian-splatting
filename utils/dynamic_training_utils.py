import numpy as np
import cv2
import torch

from scene.cameras import Camera
from scene import Scene, GaussianModel

def project_gaussians_to_image(gaussians: GaussianModel, viewpoint_cam: Camera):
    gt_image = viewpoint_cam.original_image.cpu()
    gt_image_np = convert_cpu_image_to_numpy(gt_image)
    camera_r = torch.tensor(viewpoint_cam.R.T, dtype=torch.float32).cpu()
    camera_t = torch.tensor(viewpoint_cam.T, dtype=torch.float32).reshape(3, 1).cpu()
    intrinsic = torch.tensor(viewpoint_cam.K, dtype=torch.float32).cpu()

    gaussians_xyz = gaussians.get_xyz.cpu()
    gaussians_xyz = gaussians_xyz.T

    xyzs_cam = torch.matmul(camera_r, gaussians_xyz) + camera_t
    xyzs_cam = xyzs_cam.T

    points_2d = torch.matmul(xyzs_cam, intrinsic.T)
    points_2d = points_2d[:, :2] / points_2d[:, 2:]

    points_2d = points_2d.numpy().astype(np.int32)

    if gt_image_np is not None:
        for (u, v) in points_2d:
            cv2.circle(gt_image_np, (int(u), int(v)), 1, (0, 255, 0), -1)
        cv2.imwrite("projected_gaussians.png", gt_image_np)
        
    return 


def print_using_variable(iteration, viewpoint_cam: Camera, gaussians: GaussianModel, rendered_image):
    print("[print_using_variable] Iteration:", iteration)

    # print viewpoint_cam and gt_image
    print("[print_using_variable] Viewpoint Camera:", viewpoint_cam.__dict__.keys())
    print("[print_using_variable] Gaussians:", gaussians.__dict__.keys())
    
    # print gaussians xyz, features, opacity, scales, rotations shape
    print("[print_using_variable] Gaussians XYZ shape:", gaussians.get_xyz.shape)
    print("[print_using_variable] Gaussians Features shape:", gaussians.get_features.shape)
    print("[print_using_variable] Gaussians Opacity shape:", gaussians.get_opacity.shape)
    print("[print_using_variable] Gaussians Rotations shape:", gaussians.get_rotation.shape)

    gt_image = viewpoint_cam.original_image.cpu()
    # display gt image
    gt_image_np = convert_cpu_image_to_numpy(gt_image)

    # display rendered image
    rendered_image_np = convert_cpu_image_to_numpy(rendered_image.cpu().detach())
    if gt_image_np is not None and rendered_image_np is not None:
        combined_image = np.hstack((gt_image_np, rendered_image_np))
        cv2.imwrite("combined_image_{}.png".format(iteration), combined_image)

def convert_cpu_image_to_numpy(cpu_image):
    if cpu_image is not None and hasattr(cpu_image, 'shape'):
        image_trans = cpu_image.numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format
        image_np = image_trans[..., [2, 1, 0]]  # Convert RGB to BGR for OpenCV
        image_np = np.ascontiguousarray(image_np)  # Ensure contiguous memory layout
        image_np = (image_np * 255.0).astype('uint8')  # Convert to uint8 for OpenCV
        return image_np
    return None