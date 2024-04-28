//
// Created by Amit on 28/04/2024.
//

#ifndef ONNX_INFERENCE_EXAMPLE_INFERENCE_H
#define ONNX_INFERENCE_EXAMPLE_INFERENCE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <libproc.h>
#include <unistd.h>

#include "onnxruntime/core/session/onnxruntime_c_api.h"

#define IMAGE_W     28
#define IMAGE_H     28
#define NUM_CLASSES 10

const OrtApi *ort_api;

#define ORT_ABORT_ON_ERROR(expr)                                \
  do {                                                          \
    OrtStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                  \
      const char* msg = ort_api->GetErrorMessage(onnx_status);  \
      fprintf(stderr, "%s\n", msg);                             \
      ort_api->ReleaseStatus(onnx_status);                      \
      abort();                                                  \
    }                                                           \
  } while (0);

typedef struct {
    OrtSession *session;
    OrtSessionOptions *options;
    OrtEnv *env;
} ort_session_t;

int set_ort_api() {
    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    if (!ort_api) {
        printf("Failed to initialise OnnxRuntime\n");
        return -1;
    }

    return 0;
}

ort_session_t *get_ort_session() {
    const char *model_path = "model.onnx";
    OrtEnv *ort_env;
    OrtSessionOptions* ort_session_options;
    OrtSession *ort_session;

    ORT_ABORT_ON_ERROR(ort_api->CreateEnv(ORT_LOGGING_LEVEL_INFO, "test", &ort_env));
    ORT_ABORT_ON_ERROR(ort_api->CreateSessionOptions(&ort_session_options));
    ORT_ABORT_ON_ERROR(ort_api->CreateSession(ort_env, model_path, ort_session_options, &ort_session));

    ort_session_t *session = (ort_session_t *) malloc(sizeof(ort_session_t));

    session->session = ort_session;
    session->options = ort_session_options;
    session->env = ort_env;

    return session;
}

void run_inference(OrtSession *session, const uint8_t *inference_image, float *scores) {
    OrtMemoryInfo *memory_info;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    size_t model_input_size = sizeof(float) * IMAGE_H * IMAGE_W;
    int64_t input_shape[] = {1, IMAGE_H, IMAGE_W};
    const char* input_names[] = {"model_input_input"};
    const char* output_names[] = {"model_output"};

    float *model_input = (float *) malloc(model_input_size);
    float *model_output = NULL;

    for (uint8_t i = 0; i < IMAGE_H; i++) {
        for (uint8_t j = 0; j < IMAGE_W; j++) {
            model_input[i * IMAGE_W + j] = (float) inference_image[i * IMAGE_W + j] / 255.0f;
            // printf("%f,", model_input[i * IMAGE_W + j]);
        }
    }

    // printf("\n");

    ORT_ABORT_ON_ERROR(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeCPU, &memory_info));
    ORT_ABORT_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
            memory_info,
            model_input,
            model_input_size,
            input_shape,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor
    ));

    ORT_ABORT_ON_ERROR(ort_api->Run(
            session,
            NULL,
            input_names,
            (const OrtValue* const*)&input_tensor,
            1,
            output_names,
            1,
            &output_tensor
    ));

    ORT_ABORT_ON_ERROR(ort_api->GetTensorMutableData(output_tensor, (void**)&model_output));

    for (size_t i = 0; i < NUM_CLASSES; i++) {
        scores[i] = model_output[i];
    }

    ort_api->ReleaseMemoryInfo(memory_info);
    ort_api->ReleaseValue(output_tensor);
    ort_api->ReleaseValue(input_tensor);

    free(model_input);
    model_input = NULL;
}

void release_ort_session(ort_session_t *ort_session) {
    ort_api->ReleaseSessionOptions(ort_session->options);
    ort_api->ReleaseSession(ort_session->session);
    ort_api->ReleaseEnv(ort_session->env);

    free(ort_session);
    ort_session = NULL;
}

#endif //ONNX_INFERENCE_EXAMPLE_INFERENCE_H
