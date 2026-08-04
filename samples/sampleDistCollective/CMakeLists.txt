#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if(${TRT_BUILD_SAMPLES_LINK_STATIC_TRT})
    message(FATAL_ERROR "sampleDistCollective does not support linking with static TensorRT libraries. Please set TRT_BUILD_SAMPLES_LINK_STATIC_TRT to OFF.")
endif()

include(InstallImportedLibraries)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/modules/FindNCCL.cmake")
    return()
endif()

# On Linux, NCCL is a build-time conan dep and gets linked + bundled with
# libnvinfer. On Windows it is not a build-time dep (the Windows nccl conan
# package's package_id resolution fights with the build profile's cuda
# override), so the sample uses LoadLibrary at runtime and nccl.dll is
# fetched at test time via test_deps.py.
if(NOT WIN32 AND NOT DEFINED NCCL_LIB_DIRS)
    return()
endif()

if(NOT WIN32)
    find_package(NCCL REQUIRED)
endif()

add_executable(sample_dist_collective
    sampleDistCollective.cpp
)

if(NOT WIN32)
    installImportedLibraries(
        TARGETS NCCL::nccl
        COMPONENT internal
    )
endif()

target_link_libraries(sample_dist_collective PRIVATE
    trt_samples_common
    TRT_SAMPLES::tensorrt
    $<$<NOT:$<PLATFORM_ID:Windows>>:NCCL::nccl>
    $<$<NOT:$<PLATFORM_ID:Windows>>:dl>
)
add_dependencies(tensorrt_samples sample_dist_collective)

installLibraries(
    TARGETS sample_dist_collective
    OPTIONAL
    COMPONENT internal
)
