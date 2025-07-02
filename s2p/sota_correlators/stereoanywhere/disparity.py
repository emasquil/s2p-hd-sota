import argparse
from types import SimpleNamespace
import os

import iio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from s2p import common
from s2p.sota_correlators.stereoanywhere.models.depth_anything_v2 import get_depth_anything_v2
from s2p.sota_correlators.stereoanywhere.models.stereoanywhere import StereoAnywhere


def load_stereo_and_mono_models(
    stereo_ckpt: str,
    mono_ckpt: str,
    args: SimpleNamespace,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load StereoAnywhere and Depth-Anything-v2 models for inference (single GPU only).

    Args:
        stereo_ckpt (str): Path to the StereoAnywhere checkpoint.
        mono_ckpt (str): Path to the Depth-Anything-v2 checkpoint.
        args (SimpleNamespace): Arguments needed to initialize StereoAnywhere.
        device (torch.device): Device to load models on.

    Returns:
        tuple: (stereo_model, mono_model) both on eval() mode.
    """
    print(f"USING DEVICE: {device}")
    # Load StereoAnywhere
    stereo_model = StereoAnywhere(args).to(device)
    state = torch.load(stereo_ckpt, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    stereo_model.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()}, strict=True
    )
    stereo_model.eval()

    # Load DepthAnythingV2
    mono_model = get_depth_anything_v2(mono_ckpt).to(device)
    mono_model.eval()

    return stereo_model, mono_model


def read_image(path: str) -> torch.Tensor:
    """
    Load PNG/JPG **or** GeoTIFF and return [1, 3, H, W] float32 in [0 .. 1].

    • 3-band → RGB
    • 1-band (panchromatic) → replicate channel → grey RGB
    """
    img = iio.read(path)  # [H, W, C]
    # If there are NaNs, raise a Warning and replace them with 0.0
    if np.isnan(img).any():
        print(f"Warning: NaNs found in {path}. Replacing with 0.0")
        img = np.nan_to_num(img, nan=0.0)
    # If single channel but 3D, assume it's grayscale and remove the channel dimension
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)  # remove the last dimension if it's 1
    # Dimension handling
    if img.ndim == 2:  # single channel (grayscale)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] > 3:  # more than 3 channels
        img = img[:, :, :3]
    # Pixel range normalization
    if img.max() == 255:
        img = img.astype(np.float32) / 255.0
    # elif img.max() > 1.0:
    else:
        img = (img - img.min()) / (img.max() - img.min())

    tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [3,H,W]
    return tensor.unsqueeze(0)  # [1,3,H,W]


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    """Symmetric replicate-pad so H & W are divisible by *multiple*."""
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]  # l r t b
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad):
    _, _, h, w = x.shape
    return x[..., pad[2] : h - pad[3], pad[0] : w - pad[1]]


@torch.no_grad()
def run(
    left_path, right_path, mono_ckpt, stereo_ckpt, disparity_path, mask_path
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_image = read_image(left_path).to(device)
    right_image = read_image(right_path).to(device)

    # Load models
    args = SimpleNamespace(
        maxdisp=192,
        n_downsample=2,
        n_additional_hourglass=0,
        volume_channels=8,
        vol_downsample=0,
        vol_n_masks=8,
        use_truncate_vol=False,
        mirror_conf_th=0.98,
        mirror_attenuation=0.9,
        use_aggregate_stereo_vol=False,
        use_aggregate_mono_vol=False,
        iters=32,
    )
    stereo_model, mono_model = load_stereo_and_mono_models(stereo_ckpt, mono_ckpt, args)

    # Monocular inference
    cat = torch.cat([left_image, right_image], dim=0)  # [2, 3, H, W]
    mono_depths = mono_model.infer_image(
        cat, input_size_width=cat.shape[-1], input_size_height=cat.shape[-2]
    )  # [2, 1, H, W]
    # normalize per pair 0-1
    mono_depths = (mono_depths - mono_depths.min()) / (
        mono_depths.max() - mono_depths.min()
    )
    mono_left, mono_right = mono_depths[0:1], mono_depths[1:2]

    # Pad everything to multiple of 32
    left_image, pad_left = pad_to_multiple(left_image)  # [1, 3, H, W]
    right_image, _ = pad_to_multiple(right_image)  # [1, 3, H, W]
    mono_left, _ = pad_to_multiple(mono_left)
    mono_right, _ = pad_to_multiple(mono_right)
    # Stereo inference
    with torch.no_grad():
        disparity, _ = stereo_model(
            left_image,
            right_image,
            mono_left,
            mono_right,
            test_mode=True,
            iters=stereo_model.args.iters,
        )
        # disparity = -disparity
        disparity = unpad(disparity, pad_left).squeeze([0, 1]).cpu().numpy()

    # Save disparity
    iio.write(disparity_path, disparity)

    # Rejection mask
    create_rejection_mask(disparity_path, left_path, right_path, mask_path)


def create_rejection_mask(disp_path, im1_path, im2_path, mask_path):
    im1 = common.rio_read_as_array_with_nans(im1_path)
    im2 = common.rio_read_as_array_with_nans(im2_path)
    disp = common.rio_read_as_array_with_nans(disp_path)

    if im2.ndim == 3:
        im2 = im2[0]
    if im1.ndim == 3:
        im1 = im1[0]


    h, w = disp.shape[:2]
    flow = np.stack((np.zeros_like(disp), disp), axis=-1)
    flow[..., 0] += np.arange(h)[:, None]
    flow[..., 1] += np.arange(w)

    m = ndimage.map_coordinates(
        im2, flow.transpose(2, 0, 1), order=1, mode="constant", cval=np.nan
    )
    m = (np.isfinite(im1) * np.isfinite(m) * np.isfinite(flow[..., 0])).astype(np.uint8)

    common.rasterio_write(mask_path, m)
