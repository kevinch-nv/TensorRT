/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_PLUGIN_CUFFT_WRAPPER_H
#define TRT_PLUGIN_CUFFT_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <memory>

namespace nvinfer1
{
namespace pluginInternal
{

//! Copy of the cuFFT result codes. cuFFT is loaded via dlopen, so the real
//! cufft.h is intentionally not included (it is not on the plugin include path
//! when CUDA comes from a stripped toolkit).
enum cufftResult
{
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16
};

using cufftHandle = int;

//! Transform direction constants (cuFFT defines these as macros).
constexpr int kCUFFT_FORWARD = -1;
constexpr int kCUFFT_INVERSE = 1;

//! dlopen-based wrapper over the subset of cuFFT used by FFTPlugin. Mirrors
//! CublasWrapper: no link-time dependency on libcufft.
class CufftWrapper
{
public:
    CufftWrapper();

    cufftResult cufftCreate(cufftHandle* plan) const;
    cufftResult cufftDestroy(cufftHandle plan) const;
    cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) const;
    cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) const;
    cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea) const;
    cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, int64_t* n, int64_t* inembed, int64_t istride,
        int64_t idist, cudaDataType inputType, int64_t* onembed, int64_t ostride, int64_t odist,
        cudaDataType outputType, int64_t batch, size_t* workSize, cudaDataType executionType) const;
    cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction) const;

private:
    void* tryLoadingCufft();

    //! Closes the dlopen'd cuFFT library handle owned by mLibrary.
    struct CloseLibrary
    {
        void operator()(void* handle) const;
    };
    std::unique_ptr<void, CloseLibrary> mLibrary;

    cufftResult (*_cufftCreate)(cufftHandle*){nullptr};
    cufftResult (*_cufftDestroy)(cufftHandle){nullptr};
    cufftResult (*_cufftSetAutoAllocation)(cufftHandle, int){nullptr};
    cufftResult (*_cufftSetStream)(cufftHandle, cudaStream_t){nullptr};
    cufftResult (*_cufftSetWorkArea)(cufftHandle, void*){nullptr};
    cufftResult (*_cufftXtMakePlanMany)(cufftHandle, int, int64_t*, int64_t*, int64_t, int64_t, cudaDataType, int64_t*,
        int64_t, int64_t, cudaDataType, int64_t, size_t*, cudaDataType){nullptr};
    cufftResult (*_cufftXtExec)(cufftHandle, void*, void*, int){nullptr};
};

CufftWrapper const& getCufftWrapper();

} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUFFT_WRAPPER_H
