# ONNX Inference C

Using ONNX runtime to run inference on a ONNX mdoel in C.

![ONNX in C](./assets/onnx_in_c.gif)

## Requirements

- C Compiler
- [Task](https://taskfile.dev) or user-defined build system
- Raylib
- ONNX Runtime
- Conda

I have provided a `Taskfile` to help setup the requirements on macOS quickly with: `task setup`

## Usage

- Install dependencies and setup conda environment: `task setup`
- Train the model and convert it to ONNX: `task train`
- Build and run the app: `task`
