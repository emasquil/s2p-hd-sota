import json
import os
import subprocess

import rpcm


def rpc_from_json(json_path, return_dict=False):
    with open(json_path) as f:
        d = json.load(f)
    if return_dict:
        return d["rpc"]
    return rpcm.RPCModel(d["rpc"], dict_format="rpcm")


def aoi_from_json(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d["geojson"]


def run_s2p(img_path1, img_path2, json_path1, json_path2, s2p_out_dir, horizontal_translation_margin=0):
    # define s2p config
    config = {
        "images": [
            {"img": img_path1, "rpc": rpc_from_json(json_path1, return_dict=True)},
            {"img": img_path2, "rpc": rpc_from_json(json_path2, return_dict=True)},
        ],
        "out_dir": ".",
        "dsm_resolution": 0.5,
        "roi_geojson": aoi_from_json(json_path1),
        "debug": False,
        "rectification_method": "sift",
        # Settings for the stereoanywhere correlator
        "matching_algorithm": "stereoanywhere",
        "stereo_ckpt": "/mnt/cdisk/emasquil/pretrained_models/stereoanywhere/sceneflow.tar",
        "mono_ckpt": "/mnt/cdisk/emasquil/pretrained_models/depth-anything-v2/depth_anything_v2_vitl.pth",
        "disp_range_flag": "negative",
        "register_with_shear": True,
        "horizontal_margin": 0,
        "vertical_margin": 0,
        "disp_range_extra_margin": 0,
        "horizontal_translation_margin": horizontal_translation_margin,  # Margin to add to the horizontal translation for disparity range adjustment
    }

    # write s2p config to disk
    os.makedirs(s2p_out_dir, exist_ok=True)
    config_path = os.path.join(s2p_out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # run s2p and redirect output to log file
    log_file = os.path.join(s2p_out_dir, "log.txt")
    out_dsm_path = os.path.join(s2p_out_dir, "dsm.tif")
    if not os.path.exists(out_dsm_path):
        print(f"Running s2p for {img_path1} and {img_path2} ...")
        with open(log_file, "w") as outfile:
            subprocess.run(["s2p", config_path], stdout=outfile, stderr=outfile)
    assert os.path.exists(out_dsm_path)
    print(f"... done! Output dsm: {out_dsm_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run s2p with StereoAnywhere correlator."
    )
    parser.add_argument("img_path1", type=str, help="Path to the first image.")
    parser.add_argument("img_path2", type=str, help="Path to the second image.")
    parser.add_argument(
        "json_path1", type=str, help="Path to the RPC JSON for the first image."
    )
    parser.add_argument(
        "json_path2", type=str, help="Path to the RPC JSON for the second image."
    )
    parser.add_argument(
        "s2p_out_dir", type=str, help="Output directory for s2p results."
    )
    parser.add_argument(
        "--horizontal_translation_margin", 
        type=int,
        default=0,
        help="Margin to add to the horizontal translation for disparity range adjustment."
    )

    args = parser.parse_args()
    run_s2p(
        args.img_path1,
        args.img_path2,
        args.json_path1,
        args.json_path2,
        args.s2p_out_dir,
        args.horizontal_translation_margin
    )
