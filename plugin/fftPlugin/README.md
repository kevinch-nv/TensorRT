# FFTPlugin

**Table Of Contents**
- [Description](#description)
- [Structure](#structure)
    * [Inputs](#inputs)
    * [Outputs](#outputs)
    * [Parameters](#parameters)
    * [Complex layout and normalization](#complex-layout-and-normalization)
- [Additional resources](#additional-resources)
- [License](#license)

## Description

`FFTPlugin` computes a Fast Fourier Transform with [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html). It transforms the trailing `ndims` dimensions of the input and batches over the leading dimensions. It supports complex-to-complex (C2C), onesided real-to-complex (R2C), and onesided complex-to-real (C2R) transforms in FP32, FP16, and BF16, for 1D, 2D, and 3D signals.

The plugin backs the ONNX `DFT` operator. The ONNX parser routes `DFT` nodes to this plugin, so most users never construct the plugin directly.

## Structure

### Inputs

`FFTPlugin` takes one or two inputs.

- `input`: the signal.
    - C2C: complex, `[..., N, 2]` (last dim is `[real, imag]`).
    - R2C: real, `[..., N]`.
    - C2R: complex, `[..., N/2 + 1, 2]`.
- `fft_length` (optional, C2R only): an `int64` shape input giving the original signal length `N`. Required to reconstruct odd-length signals, since both even `N` and odd `N` map to the same `N/2 + 1` frequency bins. Marked as a shape input via the ONNX `tensorrt_plugin_shape_input_indices` attribute.

### Outputs

`FFTPlugin` produces a single output, with the same element type as `input`.

- C2C: complex, `[..., N, 2]`.
- R2C: complex, `[..., N/2 + 1, 2]`.
- C2R: real, `[..., N]`.

### Parameters

| Parameter  | Type  | Description |
|------------|-------|-------------|
| `inverse`  | int32 | `0` for the forward transform, `1` for the inverse. |
| `onesided` | int32 | `0` for C2C, `1` for the onesided R2C (forward) or C2R (inverse) transform. |
| `ndims`    | int32 | Number of trailing dimensions to transform: `1`, `2`, or `3`. |

The creator rejects values outside these ranges at plugin creation.

### Complex layout and normalization

Complex values use the interleaved `[..., 2]` (real, imaginary) layout shared by ONNX `DFT`, PyTorch's `torch.view_as_real`, and the existing TensorRT STFT importer. TensorRT does not need a first-class complex type because the layers above the plugin already pack complex as real.

The plugin follows cuFFT's **unnormalized** convention: neither the forward nor the inverse transform divides by `N`. To match PyTorch's normalized inverse, divide the inverse output by `N`.

FP16 and BF16 transforms require power-of-two signal lengths along every transformed dimension (a cuFFT restriction). The plugin rejects non-power-of-two FP16/BF16 shapes rather than producing incorrect output.

## Additional resources

- [ONNX DFT operator](https://onnx.ai/onnx/operators/onnx__DFT.html)
- [cuFFT documentation](https://docs.nvidia.com/cuda/cufft/index.html)
- [TensorRT plugin API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#plugins)

## Changelog

- 2026/07/07: Reject out-of-range `inverse` and `onesided` values at plugin creation
- 2026/06/22: Initial release of this plugin 

## Known issues

- The transform axis must be the trailing signal axis (`axis == -2`). A `DFT` on an interior axis is reported as `kUNSUPPORTED_NODE`. Constant-fold or transpose the axis first.
- FP16 and BF16 require power-of-two signal lengths along every transformed dimension (a cuFFT restriction). Non-power-of-two shapes are rejected rather than producing incorrect output.
- `dft_length` padding/truncation is not supported. For the onesided inverse (C2R) it is used only to disambiguate odd `N`.

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.
