// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteReduceScatterMinimalAsync::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis) {
    return ttnn::operations::experimental::ccl::reduce_scatter_minimal_async(
        input_tensor,
        persistent_output_buffers,
        dim,
        multi_device_global_semaphore,
        num_links,
        memory_config,
        topology,
        subdevice_id,
        cluster_axis);
}
}  // namespace ttnn::operations::experimental::ccl
