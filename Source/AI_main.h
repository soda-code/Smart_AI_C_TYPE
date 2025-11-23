#ifndef _AI_MAIN // include guard for 3rd party interop
#define _AI_MAIN
#include <string>

using namespace std; 

#define NUM_LAYERS 10 
// 学习率
#define LEARNING_RATE 0.01 
#define EPOCHS 20000 

extern const float B_8[1];
// Layer 结构体
typedef struct {
    float* weights; // 权重矩阵 (InputSize * OutputSize)
    float* biases;  // 偏置向量 (OutputSize)

    float* outputs; // 这一层神经元的输出 (OutputSize)
    float* inputs;  // 这一层神经元的输入 (InputSize) - 用于反向传播
    float* deltas;  // 误差项 (OutputSize)

    int input_size;
    int output_size;
} Layer1;

extern Layer1* layer;
extern Layer1 network[NUM_LAYERS - 1];

extern  int LAYER_SIZES[NUM_LAYERS];
void setup_dummy_network(void);
void save_weights_to_c_file_cpp(const std::string& filename);
#endif