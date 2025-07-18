// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::detail {

tt::tt_metal::operation::ProgramWithCallbacks padded_slice_multi_core(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step);

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor);
}  // namespace ttnn::operations::experimental::detail
