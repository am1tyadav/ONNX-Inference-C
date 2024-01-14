#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <libproc.h>
#include <unistd.h>

#include "onnxruntime/onnxruntime_c_api.h"
#include "raylib.h"

#define SCREEN_W    896
#define SCREEN_H    448
#define TARGET_FPS  60
#define IMAGE_W     28
#define IMAGE_H     28
#define PIXEL_SIZE  16
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

uint8_t inference_image[IMAGE_H][IMAGE_W];
bool is_predicting = false;
float scores[10];

void initialise_image() {
    for (uint8_t i = 0; i < IMAGE_H; i++) {
        for (uint8_t j = 0; j < IMAGE_W; j++) {
            inference_image[i][j] = 0;
        }
    }
}

void run_inference(OrtSession *session) {
    OrtMemoryInfo *memory_info;
    OrtValue *input_tensor;
    OrtValue *output_tensor;
    size_t model_input_size = sizeof(float) * IMAGE_H * IMAGE_W;
    int64_t input_shape[] = {1, IMAGE_H, IMAGE_W};
    const char* input_names[] = {"model_input_input"};
    const char* output_names[] = {"model_output"};

    float *model_input = (float *) malloc(model_input_size);
    float *model_output = NULL;

    for (uint8_t i = 0; i < IMAGE_H; i++) {
        for (uint8_t j = 0; j < IMAGE_W; j++) {
            model_input[i * IMAGE_W + j] = (float) inference_image[i][j] / 255.0f;
        }
    }

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

int main() {
    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    if (!ort_api) {
        printf("Failed to initialise OnnxRuntime\n");
        return -1;
    }

    const char *model_path = "model.onnx";
    OrtEnv *ort_env;
    OrtSessionOptions* ort_session_options;
    OrtSession *ort_session;

    ORT_ABORT_ON_ERROR(ort_api->CreateEnv(ORT_LOGGING_LEVEL_INFO, "test", &ort_env));
    ORT_ABORT_ON_ERROR(ort_api->CreateSessionOptions(&ort_session_options));
    ORT_ABORT_ON_ERROR(ort_api->CreateSession(ort_env, model_path, ort_session_options, &ort_session));

    InitWindow(SCREEN_W, SCREEN_H, "MNIST Inference");
    SetTargetFPS(TARGET_FPS);

    // initialise_image();

    while (!WindowShouldClose()) {
        // Handle Inputs

        Vector2 mouse_position = GetMousePosition();

        uint8_t col = (uint8_t) (mouse_position.x / PIXEL_SIZE);
        uint8_t row = (uint8_t) (mouse_position.y / PIXEL_SIZE);

        if (col < IMAGE_W && row < IMAGE_H) {
            if (IsMouseButtonDown(0)) {
                inference_image[row][col] = 250;
            }
            if (IsMouseButtonDown(1)) {
                inference_image[row][col] = 0;
            }
        }

        if (IsKeyPressed(KEY_SPACE)) {
            is_predicting = !is_predicting;
        }
        if (IsKeyPressed(KEY_R)) {
            initialise_image();
        }

        if (is_predicting) run_inference(ort_session);

        // Draw

        BeginDrawing();

        ClearBackground(DARKGRAY);

        // Image Canvas
        for (uint8_t i = 0; i < IMAGE_H; i++) {
            for (uint8_t j = 0; j < IMAGE_W; j++) {
                uint8_t pixel = inference_image[i][j];
                DrawRectangle(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, (Color) {pixel, pixel, pixel, 255});
            }
        }

        DrawText("Draw a digit on the canvas on the left", 462, 40, 20, LIGHTGRAY);
        DrawText("Press [SPACE] to toggle inference", 462, 80, 20, LIGHTGRAY);
        DrawText("Press [R] to reset canvas", 462, 120, 20, LIGHTGRAY);
        DrawText("Press [ESC] to exit", 462, 160, 20, LIGHTGRAY);

        char fps_label[10];
        sprintf(fps_label, "FPS: %d", GetFPS());
        DrawText(fps_label, 800, 400, 20, LIGHTGRAY);

        if (is_predicting) {
            DrawText("Prediction:", 462, 200, 20, LIGHTGRAY);

            for (uint8_t i = 0; i < NUM_CLASSES; i++) {
                uint8_t c = (uint8_t) (255 * scores[i]);
                char label[2];
                sprintf(label, "%d", i);

                DrawRectangle(462 + 35 * i, 240, 30, 30, (Color) { 0, c, 0, 255 });
                DrawText(label, 468 + 35 * i, 280, 30, (Color) { 0, c, 0, 255 });
            }
        }

        EndDrawing();
    }

    ort_api->ReleaseSessionOptions(ort_session_options);
    ort_api->ReleaseSession(ort_session);
    ort_api->ReleaseEnv(ort_env);

    CloseWindow();
    return 0;
}
