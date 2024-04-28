
#include "inference.h"
#include "raylib.h"

#define SCREEN_W    448
#define SCREEN_H    896
#define TARGET_FPS  60
#define PIXEL_SIZE  16

void init_image(uint8_t *inference_image) {
    for (uint8_t i = 0; i < IMAGE_H; i++) {
        for (uint8_t j = 0; j < IMAGE_W; j++) {
            inference_image[i * IMAGE_W + j] = 0;
        }
    }
}

void handle_input(uint8_t *inference_image)
{
    Vector2 mouse_position = GetMousePosition();

    uint8_t col = (uint8_t)(mouse_position.x / PIXEL_SIZE);
    uint8_t row = (uint8_t)(mouse_position.y / PIXEL_SIZE);

    if (col < IMAGE_W && row < IMAGE_H)
    {
        if (IsMouseButtonDown(0))
        {
            inference_image[row * IMAGE_W + col] = 250;
        }
        if (IsMouseButtonDown(1))
        {
            inference_image[row * IMAGE_W + col] = 0;
        }
    }

    if (IsKeyPressed(KEY_R))
    {
        init_image(inference_image);
    }
}

void draw_everything(const uint8_t *inference_image, const float *scores)
{
    BeginDrawing();

    ClearBackground(DARKGRAY);

    // Image Canvas
    for (uint8_t i = 0; i < IMAGE_H; i++)
    {
        for (uint8_t j = 0; j < IMAGE_W; j++)
        {
            uint8_t pixel = inference_image[i * IMAGE_W + j];
            DrawRectangle(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, (Color){pixel, pixel, pixel, 255});
        }
    }

    DrawText("Draw a digit on the canvas above", 20, 40 + SCREEN_W, 20, LIGHTGRAY);
    DrawText("Press [R] to reset canvas", 20, 80 + SCREEN_W, 20, LIGHTGRAY);
    DrawText("Press [ESC] to exit", 20, 120 + SCREEN_W, 20, LIGHTGRAY);

    DrawText("Prediction:", 20, 200 + SCREEN_W, 20, LIGHTGRAY);

    for (uint8_t i = 0; i < NUM_CLASSES; i++)
    {
        uint8_t c = (uint8_t)(255 * scores[i]);
        char label[2];
        sprintf(label, "%d", i);

        DrawRectangle(20 + 35 * i, 240 + SCREEN_W, 30, 30, (Color){0, c, 0, 255});
        DrawText(label, 24 + 35 * i, 280 + SCREEN_W, 30, (Color){0, c, 0, 255});
    }

    EndDrawing();
}

int main() {
    int result = set_ort_api();

    if (result !=0) return result;

    ort_session_t *ort_session = get_ort_session();
    uint8_t inference_image[IMAGE_H * IMAGE_W];
    float scores[10];

    InitWindow(SCREEN_W, SCREEN_H, "MNIST Inference");
    SetTargetFPS(TARGET_FPS);

    init_image(inference_image);

    while (!WindowShouldClose()) {
        handle_input(inference_image);

        run_inference(ort_session->session, inference_image, scores);

        draw_everything(inference_image, scores);
    }

    release_ort_session(ort_session);
    CloseWindow();

    return 0;
}
