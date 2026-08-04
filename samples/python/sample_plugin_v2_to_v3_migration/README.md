# Migrating a Python Plugin from IPluginV2DynamicExt to IPluginV3

## Description

This sample, `sample_plugin_v2_to_v3_migration`, demonstrates how to migrate from TensorRT V2 plugins to V3 by two elementwise scale (`Y = scale * X`) implementations, one with the deprecated `IPluginV2DynamicExt` interface and one with the `IPluginV3` interface. The two implementations sit side by side so you can read the migration off directly. `IPluginV2DynamicExt` is deprecated since TensorRT 8.5 and is scheduled for removal in TensorRT 12.x, so existing plugins should move to `IPluginV3`.

### The plugin and its creator

A plugin needs two classes: the plugin itself and a creator that the plugin registry uses to build instances.

- V2: `ScalePluginV2(trt.IPluginV2DynamicExt)` with `ScalePluginV2Creator(trt.IPluginCreator)`.
- V3: `ScalePluginV3(trt.IPluginV3, trt.IPluginV3OneCore, trt.IPluginV3OneBuild, trt.IPluginV3OneRuntime)` with `ScalePluginV3Creator(trt.IPluginCreatorV3One)`.

In V3 the single plugin class is split across the base `IPluginV3` interface plus three [capability interfaces](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html#ipluginv3) (core, build, runtime). `get_capability_interface()` hands TensorRT the right interface for each phase. Here one object implements all of them, so it returns `self`.

### Method-by-method mapping

| Concern | `IPluginV2DynamicExt` (before) | `IPluginV3` (after) |
|---|---|---|
| Base classes | `IPluginV2DynamicExt` | `IPluginV3` + `IPluginV3OneCore` + `IPluginV3OneBuild` + `IPluginV3OneRuntime` |
| Creator base | `IPluginCreator` | `IPluginCreatorV3One` |
| Name attribute | `plugin_type` | `plugin_name` |
| Capability dispatch | not needed | `get_capability_interface(type)` |
| Output datatype | `get_output_datatype(index, input_types)` (one at a time) | `get_output_data_types(input_types)` (returns a list) |
| Output shape | `get_output_dimensions(index, inputs, expr_builder)` (one at a time) | `get_output_shapes(inputs, shape_inputs, expr_builder)` (returns a list) |
| Format support | `supports_format_combination(pos, in_out, num_inputs)`, `in_out[pos]` is a `PluginTensorDesc` | `supports_format_combination(pos, in_out, num_inputs)`, `in_out[pos]` is a `DynamicPluginTensorDesc` (use `in_out[pos].desc`) |
| Resource lifecycle | `initialize()` / `terminate()` | acquire in `configure_plugin()` / `on_shape_change()`. No initialize/terminate |
| Per-context setup | `attach_to_context()` / `detach_from_context()` | `attach_to_context(context)` returns the per-context clone |
| Serialization | plugin implements `serialize()` returning bytes. Creator implements `deserialize_plugin()` | plugin implements `get_fields_to_serialize()`. TensorRT serializes and re-creates through the creator. No `serialize()` or `deserialize_plugin()`. |
| Create signature | `create_plugin(name, fc)` | `create_plugin(name, fc, phase)` |
| Add to network | `network.add_plugin_v2(inputs, plugin)` | `network.add_plugin_v3(inputs, shape_inputs, plugin)` |
| `enqueue()` | unchanged | unchanged |

Note that `configure_plugin()` receives `DynamicPluginTensorDesc` in both V2 and V3. Only `supports_format_combination()` differs: V2 passes `PluginTensorDesc` while V3 passes `DynamicPluginTensorDesc`.

The biggest practical simplification is serialization. In V2 you hand-roll `serialize()` and `deserialize_plugin()`. In V3 you list the attributes to serialize in `get_fields_to_serialize()` and TensorRT rebuilds the plugin by calling the creator's `create_plugin()` with those fields, so the same attribute-parsing code serves both initial construction and deserialization.

See also [Side-by-Side V2 ↔ V3 API Mapping](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html#v2-to-v3-api-mapping)

## Running the sample

1. Prerequisites:

    ```bash
    pip3 install -r requirements.txt
    export CUDA_PATH=/usr/local/cuda
    ```

2. Run the sample. It builds and runs the op as both a V2 and a V3 plugin on the same input and checks that they produce identical outputs, which is the migration-correctness check:

    ```bash
    python3 plugin_v2_to_v3_migration.py
    ```

3. On success you should see:

    ```text
    [v2] output matches the reference (scale * X)
    [v3] output matches the reference (scale * X)
    [both] V2 and V3 produce identical outputs
    Inference result correct!
    ```

## Additional resources

The following resources give more detail on the V2 and V3 plugin interfaces:

**TensorRT plugins**
- [IPluginV3 API description (core, build, and runtime capability interfaces)](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html#ipluginv3)
- [Migrating V2 plugins to IPluginV3](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html#migrating-plugins)
- [Adding custom layers using the Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-python.html)

**Related samples**
- [Python-based NonZero Plugin (IPluginV3)](../non_zero_plugin/)
- [Python-based TRT Plugins (IPluginV2DynamicExt and IPluginV3)](../python_plugin/)

**Other documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog

- June 2026: Initial release of this sample.

## Known issues

There are no known issues in this sample.
