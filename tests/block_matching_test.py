import os
import subprocess

import pytest

import s2p
from s2p.config import get_default_config
from s2p.gpu_memory_manager import GPUMemoryManager
from tests_utils import data_path


@pytest.fixture
def gpu_memory_manager() -> GPUMemoryManager:
    return GPUMemoryManager.make_unbounded()


def test_compute_disparity_map_timeout(gpu_memory_manager, timeout=1):
    """
    Run a long call to compute_disparity_map to check that the timeout kills it.
    """
    cfg = get_default_config()
    img = data_path(os.path.join("input_pair", "img_01.tif"))
    disp = data_path(os.path.join("testoutput", "d.tif"))
    mask = data_path(os.path.join("testoutput", "m.tif"))

    with pytest.raises(subprocess.TimeoutExpired):
        s2p.block_matching.compute_disparity_map(cfg, img, img, disp, mask,
                                                 "mgm_multi", -100, 100,
                                                 timeout, gpu_mem_manager=gpu_memory_manager)


def test_compute_disparity_map_max_disp_range(gpu_memory_manager, max_disp_range=10):
    """
    Run a call to compute_disparity_map with a small max_disp_range
    to check that an error is raised.
    """
    cfg = get_default_config()
    img = data_path(os.path.join("input_pair", "img_01.tif"))
    disp = data_path(os.path.join("testoutput", "d.tif"))
    mask = data_path(os.path.join("testoutput", "m.tif"))

    with pytest.raises(s2p.block_matching.MaxDisparityRangeError):
        s2p.block_matching.compute_disparity_map(cfg, img, img, disp, mask,
                                                 "mgm_multi", -100, 100,
                                                 max_disp_range=max_disp_range,
                                                 gpu_mem_manager=gpu_memory_manager)
