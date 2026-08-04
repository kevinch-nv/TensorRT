#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorrt as trt

from polygraphy.backend.trt import CreateConfig, TrtRunner, create_network, engine_from_network
from polygraphy.json import from_json, to_json

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from plugin_utils import KernelHelper, cuda_call, volume

# The V2 and V3 plugins register under distinct names so both can coexist in the plugin
# registry when this sample runs them together.
PLUGIN_NAME_V2 = "ScalePluginV2"
PLUGIN_NAME_V3 = "ScalePluginV3"
PLUGIN_VERSION = "1"

# The kernel is identical for both plugin versions: Y[i] = scale * X[i].
scale_kernel_float = r"""
extern "C" __global__
void scale_float(float const* X, float* Y, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
    {
        Y[i] = scale * X[i];
    }
}
"""


def launch_scale_kernel(
    cu_device: int,
    input_desc: list[trt.PluginTensorDesc],
    inputs: list[int],
    outputs: list[int],
    scale: float,
    stream: int,
) -> None:
    """Launch the elementwise scale kernel."""
    n = volume(input_desc[0].dims)

    block_size = 256
    num_blocks = int((n + block_size - 1) // block_size)

    d_in = np.array([inputs[0]], dtype=np.uint64)
    d_out = np.array([outputs[0]], dtype=np.uint64)
    args = [d_in, d_out, np.array([scale], dtype=np.float32), np.array([n], dtype=np.int32)]
    kernel_args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    helper = KernelHelper(scale_kernel_float, int(cu_device))
    fn = helper.getFunction(b"scale_float")

    cuda_call(
        cuda.cuLaunchKernel(fn, num_blocks, 1, 1, block_size, 1, 1, 0, int(stream), kernel_args, 0)
    )


# ScalePlugin that uses the deprecated IPluginV2DynamicExt interface
class ScalePluginV2(trt.IPluginV2DynamicExt):
    def __init__(self, fc: trt.PluginFieldCollection | None = None) -> None:
        trt.IPluginV2DynamicExt.__init__(self)
        # V2 uses `plugin_type`. The creator must report the same string from get_plugin_name().
        self.plugin_type = PLUGIN_NAME_V2
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = 1.0
        self.cu_device: int | None = None

        if fc is not None:
            for f in fc:
                if f.name == "scale":
                    self.scale = float(f.data[0])

    def get_output_datatype(self, index: int, input_types: list[trt.DataType]) -> trt.DataType:
        return input_types[0]

    # V2 reports output shapes one output at a time, returning a DimsExprs.
    def get_output_dimensions(
        self, output_index: int, inputs: list[trt.DimsExprs], expr_builder: trt.IExprBuilder
    ) -> trt.DimsExprs:
        # Since this op is elementwise, we report the output shape same as the input.
        return trt.DimsExprs(inputs[0])

    # V2 has an explicit initialize()/terminate() resource lifecycle.
    def initialize(self) -> int:
        self.cu_device = cuda_call(cuda.cuDeviceGet(0))
        return 0

    def terminate(self) -> None:
        pass

    def configure_plugin(
        self, inp: list[trt.DynamicPluginTensorDesc], out: list[trt.DynamicPluginTensorDesc]
    ) -> None:
        pass

    # In V2, in_out elements are PluginTensorDesc (access .type / .format directly).
    def supports_format_combination(
        self, pos: int, in_out: list[trt.PluginTensorDesc], num_inputs: int
    ) -> bool:
        assert num_inputs == 1
        desc = in_out[pos]
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return desc.type == trt.DataType.FLOAT
        return in_out[0].type == desc.type

    def get_workspace_size(
        self, input_desc: list[trt.PluginTensorDesc], output_desc: list[trt.PluginTensorDesc]
    ) -> int:
        return 0

    def enqueue(
        self,
        input_desc: list[trt.PluginTensorDesc],
        output_desc: list[trt.PluginTensorDesc],
        inputs: list[int],
        outputs: list[int],
        workspace: int,
        stream: int,
    ) -> None:
        launch_scale_kernel(self.cu_device, input_desc, inputs, outputs, self.scale, stream)

    def serialize(self) -> bytes:
        return to_json({"scale": self.scale})

    def clone(self) -> ScalePluginV2:
        cloned = ScalePluginV2()
        cloned.__dict__.update(self.__dict__)
        return cloned


class ScalePluginV2Creator(trt.IPluginCreator):
    def __init__(self) -> None:
        trt.IPluginCreator.__init__(self)
        self.name = PLUGIN_NAME_V2
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("scale", np.array([]), trt.PluginFieldType.FLOAT32)]
        )

    def create_plugin(self, name: str, fc: trt.PluginFieldCollection) -> ScalePluginV2:
        return ScalePluginV2(fc)

    # V2 creators must implement deserialize_plugin() to rebuild the plugin from engine bytes.
    def deserialize_plugin(self, name: str, data: bytes) -> ScalePluginV2:
        deserialized = ScalePluginV2()
        deserialized.__dict__.update(dict(from_json(data)))
        return deserialized


# ScalePlugin using the new V3 plugin interface
# A single V2 class becomes IPluginV3 plus three capability interfaces.
# The three interfaces doesn't need to be implemented in the same class,
# but we do so here for simplicity.
class ScalePluginV3(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime):
    def __init__(self, scale: float = 1.0) -> None:
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        # V3 uses `plugin_name` (V2 used `plugin_type`).
        self.plugin_name = PLUGIN_NAME_V3
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.num_outputs = 1
        self.scale = scale
        self.cu_device: int | None = None

    # IPluginV3: hands back the capability interfaces. This object implements all three in this case.
    def get_capability_interface(self, capability_type: trt.PluginCapabilityType) -> ScalePluginV3:
        return self

    # IPluginV3OneBuild: reports all output datatypes at once.
    def get_output_data_types(self, input_types: list[trt.DataType]) -> list[trt.DataType]:
        return [input_types[0]]

    # IPluginV3OneBuild: reports all output shapes at once.
    def get_output_shapes(
        self,
        inputs: list[trt.DimsExprs],
        shape_inputs: list[trt.DimsExprs],
        expr_builder: trt.IExprBuilder,
    ) -> list[trt.DimsExprs]:
        return [trt.DimsExprs(inputs[0])]

    # IPluginV3OneBuild: in_out elements are DynamicPluginTensorDesc (access .desc.type / .desc.format).
    def supports_format_combination(
        self, pos: int, in_out: list[trt.DynamicPluginTensorDesc], num_inputs: int
    ) -> bool:
        assert num_inputs == 1
        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0:
            return desc.type == trt.DataType.FLOAT
        return in_out[0].desc.type == desc.type

    # IPluginV3OneBuild: no initialize()/terminate() in V3. Acquire resources here instead.
    def configure_plugin(
        self, inp: list[trt.DynamicPluginTensorDesc], out: list[trt.DynamicPluginTensorDesc]
    ) -> None:
        self.cu_device = cuda_call(cuda.cuDeviceGet(0))

    # IPluginV3OneRuntime: TensorRT serializes the plugin for you from these fields. No serialize()
    # on the plugin and no deserialize_plugin() on the creator are needed.
    def get_fields_to_serialize(self) -> trt.PluginFieldCollection:
        return trt.PluginFieldCollection(
            [trt.PluginField("scale", np.array([self.scale], dtype=np.float32), trt.PluginFieldType.FLOAT32)]
        )

    # IPluginV3OneRuntime
    def on_shape_change(
        self, inp: list[trt.PluginTensorDesc], out: list[trt.PluginTensorDesc]
    ) -> None:
        self.cu_device = cuda_call(cuda.cuDeviceGet(0))

    # IPluginV3OneRuntime
    def enqueue(
        self,
        input_desc: list[trt.PluginTensorDesc],
        output_desc: list[trt.PluginTensorDesc],
        inputs: list[int],
        outputs: list[int],
        workspace: int,
        stream: int,
    ) -> None:
        launch_scale_kernel(self.cu_device, input_desc, inputs, outputs, self.scale, stream)

    # IPluginV3OneRuntime: replaces the V2 attach/detach-from-context pair.
    def attach_to_context(self, context: trt.IPluginResourceContext) -> ScalePluginV3:
        return self.clone()

    # IPluginV3OneRuntime
    def set_tactic(self, tactic: int) -> None:
        pass

    # IPluginV3
    def clone(self) -> ScalePluginV3:
        cloned = ScalePluginV3()
        cloned.__dict__.update(self.__dict__)
        return cloned


class ScalePluginV3Creator(trt.IPluginCreatorV3One):
    def __init__(self) -> None:
        trt.IPluginCreatorV3One.__init__(self)
        self.name = PLUGIN_NAME_V3
        self.plugin_version = PLUGIN_VERSION
        self.plugin_namespace = ""
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("scale", np.array([]), trt.PluginFieldType.FLOAT32)]
        )

    def create_plugin(
        self, name: str, fc: trt.PluginFieldCollection, phase: trt.TensorRTPhase
    ) -> ScalePluginV3:
        scale = 1.0
        for f in fc:
            if f.name == "scale":
                scale = float(f.data[0])
        return ScalePluginV3(scale)


def build_and_run(plugin_version: str, X: np.ndarray, scale: float) -> np.ndarray:
    """Build a single-plugin engine for the given version, run it on X, and return the output."""
    plugin_name = PLUGIN_NAME_V2 if plugin_version == "v2" else PLUGIN_NAME_V3

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, namespace="")
    plg_registry = trt.get_plugin_registry()

    if plugin_version == "v2":
        plg_registry.register_creator(ScalePluginV2Creator(), "")
    else:
        plg_registry.register_creator(ScalePluginV3Creator(), "")

    builder, network = create_network(strongly_typed=True)
    input_X = network.add_input(name="X", dtype=trt.float32, shape=X.shape)

    pfc = trt.PluginFieldCollection(
        [trt.PluginField("scale", np.array([scale], dtype=np.float32), trt.PluginFieldType.FLOAT32)]
    )
    creator = plg_registry.get_creator(plugin_name, PLUGIN_VERSION, "")

    if plugin_version == "v2":
        plugin = creator.create_plugin(plugin_name, pfc)
        out = network.add_plugin_v2([input_X], plugin)
    else:
        plugin = creator.create_plugin(plugin_name, pfc, trt.TensorRTPhase.BUILD)
        out = network.add_plugin_v3([input_X], [], plugin)

    out.get_output(0).name = "Y"
    network.mark_output(out.get_output(0))

    engine = engine_from_network((builder, network), CreateConfig())
    with TrtRunner(engine, "trt_runner") as runner:
        return runner.infer({"X": X})["Y"]


def main() -> bool:
    argparse.ArgumentParser(
        description="This sample demonstrates how to migrate from TensorRT V2 plugins to V3"
    ).parse_args()

    cuda_call(cuda.cuInit(0))
    cuda_call(cudart.cudaFree(0))

    scale = 2.0
    X = np.random.normal(size=(4, 16)).astype(np.float32)
    Y_ref = (scale * X).astype(np.float32)

    # Build and run the same op as a V2 plugin and a V3 plugin on the same input.
    outputs = {}
    passed = True
    for v in ("v2", "v3"):
        outputs[v] = build_and_run(v, X, scale)
        matches_ref = np.allclose(outputs[v], Y_ref, atol=1e-2)
        print(f"[{v}] output {'matches' if matches_ref else 'does not match'} the reference (scale * X)")
        passed = passed and matches_ref

    # The whole point of the migration is that V3 reproduces V2 exactly.
    same = np.allclose(outputs["v2"], outputs["v3"], atol=1e-6)
    print(f"[both] V2 and V3 produce {'identical' if same else 'different'} outputs")
    passed = passed and same

    print("Inference result correct!" if passed else "Inference result incorrect!")
    return passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
