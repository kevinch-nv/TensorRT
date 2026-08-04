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

#include "cufftWrapper.h"
#include "common/checkMacrosPlugin.h"
#include "cudaDriverWrapper.h"

#include <string>
#include <vector>

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
#else // defined(_WIN32)
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

namespace
{
// cuFFT's SONAME major version is independent of the CUDA toolkit major (e.g.
// libcufft.so.12 ships with CUDA 13.x), so probe the known majors in order.
std::vector<std::string> cufftLibCandidates()
{
    auto const cudaMaj = std::to_string(nvinfer1::getCudaLibVersionMaj());
#if defined(_WIN32)
    return {"cufft64_" + cudaMaj + ".dll", "cufft64_12.dll", "cufft64_11.dll"};
#else
    return {"libcufft.so." + cudaMaj, "libcufft.so.12", "libcufft.so.11", "libcufft.so.10", "libcufft.so"};
#endif
}
} // namespace

namespace nvinfer1::pluginInternal
{

CufftWrapper::CufftWrapper()
    : mLibrary(tryLoadingCufft())
{
    PLUGIN_VALIDATE(mLibrary != nullptr);
    auto loadSym = [](void* handle, char const* name) {
        void* ret = dllGetSym(handle, name);
        std::string loadError = "Fail to load symbol " + std::string(name) + " from the cufft library.";
        PLUGIN_VALIDATE(ret != nullptr, loadError.c_str());
        return ret;
    };
    void* lib = mLibrary.get();
    _cufftCreate = reinterpret_cast<cufftResult (*)(cufftHandle*)>(loadSym(lib, "cufftCreate"));
    _cufftDestroy = reinterpret_cast<cufftResult (*)(cufftHandle)>(loadSym(lib, "cufftDestroy"));
    _cufftSetAutoAllocation
        = reinterpret_cast<cufftResult (*)(cufftHandle, int)>(loadSym(lib, "cufftSetAutoAllocation"));
    _cufftSetStream = reinterpret_cast<cufftResult (*)(cufftHandle, cudaStream_t)>(loadSym(lib, "cufftSetStream"));
    _cufftSetWorkArea = reinterpret_cast<cufftResult (*)(cufftHandle, void*)>(loadSym(lib, "cufftSetWorkArea"));
    _cufftXtMakePlanMany = reinterpret_cast<cufftResult (*)(cufftHandle, int, int64_t*, int64_t*, int64_t, int64_t,
        cudaDataType, int64_t*, int64_t, int64_t, cudaDataType, int64_t, size_t*, cudaDataType)>(
        loadSym(lib, "cufftXtMakePlanMany"));
    _cufftXtExec = reinterpret_cast<cufftResult (*)(cufftHandle, void*, void*, int)>(loadSym(lib, "cufftXtExec"));
}

void CufftWrapper::CloseLibrary::operator()(void* handle) const
{
    if (handle != nullptr)
    {
        dllClose(handle);
    }
}

void* CufftWrapper::tryLoadingCufft()
{
    for (auto const& name : cufftLibCandidates())
    {
        if (void* cufftLib = dllOpen(name.c_str()))
        {
            return cufftLib;
        }
    }
    PLUGIN_VALIDATE(false, "Failed to load the cuFFT library (libcufft).");
    return nullptr;
}

cufftResult CufftWrapper::cufftCreate(cufftHandle* plan) const
{
    return (*_cufftCreate)(plan);
}

cufftResult CufftWrapper::cufftDestroy(cufftHandle plan) const
{
    return (*_cufftDestroy)(plan);
}

cufftResult CufftWrapper::cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) const
{
    return (*_cufftSetAutoAllocation)(plan, autoAllocate);
}

cufftResult CufftWrapper::cufftSetStream(cufftHandle plan, cudaStream_t stream) const
{
    return (*_cufftSetStream)(plan, stream);
}

cufftResult CufftWrapper::cufftSetWorkArea(cufftHandle plan, void* workArea) const
{
    return (*_cufftSetWorkArea)(plan, workArea);
}

cufftResult CufftWrapper::cufftXtMakePlanMany(cufftHandle plan, int rank, int64_t* n, int64_t* inembed, int64_t istride,
    int64_t idist, cudaDataType inputType, int64_t* onembed, int64_t ostride, int64_t odist, cudaDataType outputType,
    int64_t batch, size_t* workSize, cudaDataType executionType) const
{
    return (*_cufftXtMakePlanMany)(plan, rank, n, inembed, istride, idist, inputType, onembed, ostride, odist,
        outputType, batch, workSize, executionType);
}

cufftResult CufftWrapper::cufftXtExec(cufftHandle plan, void* input, void* output, int direction) const
{
    return (*_cufftXtExec)(plan, input, output, direction);
}

CufftWrapper const& getCufftWrapper()
{
    static CufftWrapper sCufftWrapper;
    return sCufftWrapper;
}

} // namespace nvinfer1::pluginInternal
