# ONNX Inference In C

Using ONNX runtime to run inference on an ONNX model in C.

![ONNX in C](./assets/onnx_in_c.gif)

## Requirements

- C Compiler
- CMake
- Conda

## Usage

Setup python environment, for example, using conda:

```bash
conda create -n onnx-example python=3.10 -y

conda activate onnx-example

python -m pip install -r requirements.txt

python train.py
```

This will train and create a TF saved model in `saved_model` directory.

Next, we convert the model to onnx:

```bash
python -m tf2onnx.convert --saved-model ./saved_model --output ./model.onnx
```

Now the model is available in the ONNX format, we can build and run the app:

```bash
cmake -B ./cmake-build-debug -S .
cmake --build ./cmake-build-debug --target onnx_inference_example

./cmake-build-debug/onnx_inference_example
```

Note: Command for building ONNX for your platform will have to be explicitly set as I'm only building for macOS arm64.

For convenience, you can install [Task](https://taskfile.dev/) and run `task` in the terminal to see the list of available tasks:

```txt
task: Available tasks for this project:

* build:         Build C project
* conda:         Setup conda environment
* default:       List all tasks
* run:           Run executable
* train:         Train MNIST model and convert to ONNX format
```
