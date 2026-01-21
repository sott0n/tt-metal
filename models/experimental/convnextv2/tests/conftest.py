# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for ConvNextV2 tests

Note: Device fixture with l1_small_size is configured via @pytest.mark.parametrize
with device_params in individual test files. This follows the root conftest.py pattern.
"""

import pytest


@pytest.fixture(scope="session")
def model_config():
    """ConvNextV2-base configuration"""
    return {
        "image_size": 224,
        "patch_size": 4,
        "num_channels": 3,
        "hidden_sizes": [128, 256, 512, 1024],
        "depths": [3, 3, 27, 3],
        "num_stages": 4,
        "num_labels": 1000,
    }
