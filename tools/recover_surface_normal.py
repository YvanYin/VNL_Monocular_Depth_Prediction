import torch
import numpy as np
import torch.nn as nn


def init_image_coor(height, width):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    x = torch.from_numpy(x.copy()).cuda()
    u_u0 = x - width/2.0

    y_col = np.arange(0, height)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y[np.newaxis, :, :]
    y = y.astype(np.float32)
    y = torch.from_numpy(y.copy()).cuda()
    v_v0 = y - height/2.0
    return u_u0, v_v0


def depth_to_xyz(depth, focal_length):
    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coor(h, w)
    x = u_u0 * depth / focal_length
    y = v_v0 * depth / focal_length
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    return pw


def get_surface_normal(xyz, patch_size=5):
    # xyz: [1, h, w, 3]
    x, y, z = torch.unbind(xyz, dim=3)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y, 0)
    z = torch.unsqueeze(z, 0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    patch_weight = torch.ones((1, 1, patch_size, patch_size), requires_grad=False).cuda()
    xx_patch = nn.functional.conv2d(xx, weight=patch_weight, padding=int(patch_size / 2))
    yy_patch = nn.functional.conv2d(yy, weight=patch_weight, padding=int(patch_size / 2))
    zz_patch = nn.functional.conv2d(zz, weight=patch_weight, padding=int(patch_size / 2))
    xy_patch = nn.functional.conv2d(xy, weight=patch_weight, padding=int(patch_size / 2))
    xz_patch = nn.functional.conv2d(xz, weight=patch_weight, padding=int(patch_size / 2))
    yz_patch = nn.functional.conv2d(yz, weight=patch_weight, padding=int(patch_size / 2))
    ATA = torch.stack([xx_patch, xy_patch, xz_patch, xy_patch, yy_patch, yz_patch, xz_patch, yz_patch, zz_patch],
                      dim=4)
    ATA = torch.squeeze(ATA)
    ATA = torch.reshape(ATA, (ATA.size(0), ATA.size(1), 3, 3))
    eps_identity = 1e-6 * torch.eye(3, device=ATA.device, dtype=ATA.dtype)[None, None, :, :].repeat([ATA.size(0), ATA.size(1), 1, 1])
    ATA = ATA + eps_identity
    x_patch = nn.functional.conv2d(x, weight=patch_weight, padding=int(patch_size / 2))
    y_patch = nn.functional.conv2d(y, weight=patch_weight, padding=int(patch_size / 2))
    z_patch = nn.functional.conv2d(z, weight=patch_weight, padding=int(patch_size / 2))
    AT1 = torch.stack([x_patch, y_patch, z_patch], dim=4)
    AT1 = torch.squeeze(AT1)
    AT1 = torch.unsqueeze(AT1, 3)

    patch_num = 4
    patch_x = int(AT1.size(1) / patch_num)
    patch_y = int(AT1.size(0) / patch_num)
    n_img = torch.randn(AT1.shape).cuda()
    overlap = patch_size // 2 + 1
    for x in range(int(patch_num)):
        for y in range(int(patch_num)):
            left_flg = 0 if x == 0 else 1
            right_flg = 0 if x == patch_num -1 else 1
            top_flg = 0 if y == 0 else 1
            btm_flg = 0 if y == patch_num - 1 else 1
            at1 = AT1[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            ata = ATA[y * patch_y - top_flg * overlap:(y + 1) * patch_y + btm_flg * overlap,
                  x * patch_x - left_flg * overlap:(x + 1) * patch_x + right_flg * overlap]
            #try:
            n_img_tmp, _ = torch.solve(at1, ata)
            #except:
            #    print(at1, ata)
            n_img_tmp_select = n_img_tmp[top_flg * overlap:patch_y + top_flg * overlap, left_flg * overlap:patch_x + left_flg * overlap, :, :]
            n_img[y * patch_y:y * patch_y + patch_y, x * patch_x:x * patch_x + patch_x, :, :] = n_img_tmp_select

    #n_img, _ = torch.solve(AT1, ATA)
    n_img_L2 = torch.sqrt(torch.sum(n_img ** 2, dim=2, keepdim=True))
    n_img_norm = n_img / n_img_L2

    # re-orient normals consistently
    orient_mask = torch.sum(torch.squeeze(n_img_norm) * torch.squeeze(xyz), dim=2) > 0
    n_img_norm[orient_mask] *= -1
    return n_img_norm


def surface_normal_from_depth(depth, focal_length, valid_mask=None):
    # para depth: depth map, [b, c, h, w]
    b, c, h, w = depth.shape
    focal_length = focal_length[:, None, None, None]
    depth_filter = nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    depth_filter = nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
    xyz = depth_to_xyz(depth_filter, focal_length)
    sn_batch = []
    for i in range(b):
        xyz_i = xyz[i, :][None, :, :, :]
        normal = get_surface_normal(xyz_i)
        sn_batch.append(normal)
    sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [b, c, h, w]
    mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
    sn_batch[mask_invalid] = 0.0
    return sn_batch

def vis_normal(normal):
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    @para normal: surface normal, [h, w, 3], numpy.array
    """
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis

if __name__ == '__main__':
    import cv2, os
    import matplotlib.pyplot as plt
    from shutil import copyfile

    normal = surface_normal_from_depth(depth, focal_length=torch.tensor([512]).cuda(), valid_mask=depth>0)
    normal = normal.permute((2, 3, 1, 0)).squeeze().cpu().numpy()
    norm_vis = vis_normal(normal)
    cv2.imwrite('../../test1.png', norm_vis[:, :, ::-1])

