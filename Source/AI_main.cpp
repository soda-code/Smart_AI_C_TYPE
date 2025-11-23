#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memcpy
#include <AI_main.h> // For memcpy


// 定义每层的尺寸
 int LAYER_SIZES[NUM_LAYERS] = { 2, 8, 8, 8, 8, 8, 8, 8, 8, 1 };
// 10层：1个输入层(2), 8个隐藏层(8), 1个输出层(1)



Layer1 network[NUM_LAYERS - 1]; // 实际参数层数：9层（输入层无参数）
Layer1* layer = { 0 };

// 初始化随机浮点数 (-0.5 / sqrt(n) 到 0.5 / sqrt(n))
float random_float(int n) {
    // He initialization style scaling (for ReLU) - simpler scaling used here
    return (2.0 * ((float)rand() / RAND_MAX) - 1.0) / sqrt((float)n);
}

// ReLU Activation
float relu(float x) {
    return (x > 0.0) ? x : 0.0;
}
// ReLU Derivative
float relu_derivative(float y) {
    return (y > 0.0) ? 1.0 : 0.0;
}

// Sigmoid Activation (用于输出层)
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}
// Sigmoid Derivative (用于输出层)
float sigmoid_derivative(float y) {
    return y * (1.0 - y);
}

// 核心：通用矩阵乘法 (C = A * B)
// A: (rA x cA), B: (cA x cB), C: (rA x cB)
void matrix_multiply(const float* A, const float* B, float* C,
    int rA, int cA, int cB) {
    for (int i = 0; i < rA; i++) {
        for (int j = 0; j < cB; j++) {
            C[i * cB + j] = 0.0;
            for (int k = 0; k < cA; k++) {
                C[i * cB + j] += A[i * cA + k] * B[k * cB + j];
            }
        }
    }
}

// 通用矩阵转置乘法 (C = A * B^T)
// A: (rA x cA), B^T: (cB x rB), C: (rA x rB)
void matrix_multiply_transpose_b(const float* A, const float* B, float* C,
    int rA, int cA, int rB) {
    for (int i = 0; i < rA; i++) {
        for (int j = 0; j < rB; j++) { // j iterates over rows of B (which is B^T's columns)
            C[i * rB + j] = 0.0;
            for (int k = 0; k < cA; k++) {
                // A[i][k] * B[j][k] (row j of B, col k of B)
                C[i * rB + j] += A[i * cA + k] * B[j * cA + k];
            }
        }
    }
}

void initialize_network() {
    srand(time(NULL));

    // 实际参数层是 9 层 (从 Input->H1 到 H9->Output)
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        Layer1* layer = &network[i];

        layer->input_size = LAYER_SIZES[i];
        layer->output_size = LAYER_SIZES[i + 1];
        int weight_count = layer->input_size * layer->output_size;

        // 动态分配内存
        layer->weights = (float*)malloc(weight_count * sizeof(float));
        layer->biases = (float*)malloc(layer->output_size * sizeof(float));
        layer->outputs = (float*)malloc(layer->output_size * sizeof(float));
        layer->deltas = (float*)malloc(layer->output_size * sizeof(float));
        layer->inputs = (float*)malloc(layer->input_size * sizeof(float));

        if (!layer->weights || !layer->biases || !layer->outputs || !layer->deltas || !layer->inputs) {
            fprintf(stderr, "Error: Memory allocation failed for layer %d\n", i);
            exit(EXIT_FAILURE);
        }

        // 初始化权重和偏置
        for (int k = 0; k < weight_count; k++) {
            layer->weights[k] = random_float(layer->input_size);
        }
        for (int k = 0; k < layer->output_size; k++) {
            layer->biases[k] = 0.0; // 偏置可以初始化为0
        }
    }
}

void free_network() {
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        free(network[i].weights);
        free(network[i].biases);
        free(network[i].outputs);
        free(network[i].deltas);
        free(network[i].inputs);
    }
}

void forward_pass(const float* input, float* output) {
    float* current_input = (float*)input;
    int current_input_size = network[0].input_size;

    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        Layer1* layer = &network[i];

        // 1. 存储输入 (用于反向传播)
        memcpy(layer->inputs, current_input, layer->input_size * sizeof(float));

        // 2. 线性运算: Z = Input * W + B
        // Input: (1 x InputSize), W: (InputSize x OutputSize), Z: (1 x OutputSize)
        matrix_multiply(current_input, layer->weights, layer->outputs,
            1, layer->input_size, layer->output_size);

        // 3. 加上偏置并应用激活函数 (Outputs = Z + B)
        for (int j = 0; j < layer->output_size; j++) {
            layer->outputs[j] += layer->biases[j];

            // 最后一层使用 Sigmoid，其他使用 ReLU
            if (i == NUM_LAYERS - 2) {
                layer->outputs[j] = sigmoid(layer->outputs[j]);
            }
            else {
                layer->outputs[j] = relu(layer->outputs[j]);
            }
        }

        // 准备下一层输入
        current_input = layer->outputs;
        current_input_size = layer->output_size;
    }

    // 复制最终输出
    memcpy(output, network[NUM_LAYERS - 2].outputs, network[NUM_LAYERS - 2].output_size * sizeof(float));
}// train 函数
void train(const float* input, const float* target) {
    // 1. 获取输出层大小，并动态分配内存
    const int output_size = LAYER_SIZES[NUM_LAYERS - 1];
    float* predicted_output = (float*)malloc(output_size * sizeof(float));

    if (predicted_output == NULL) {
        fprintf(stderr, "Memory allocation error in train\n");
        exit(EXIT_FAILURE);
    }

    // 运行前向传播以获得当前预测值。
    // forward_pass 会将最终结果写入 predicted_output
    forward_pass(input, predicted_output);

    // --- 1. 计算输出层 (第 NUM_LAYERS-2 个 Layer 结构体) 的误差 (Deltas) ---
    Layer1* output_layer = &network[NUM_LAYERS - 2];
    for (int k = 0; k < output_layer->output_size; k++) {
        // 损失函数的导数 (MSE) * 激活函数的导数 (Sigmoid)
        float error = predicted_output[k] - target[k];
        output_layer->deltas[k] = error * sigmoid_derivative(predicted_output[k]);
    }

    // 释放为 predicted_output 分配的内存
    free(predicted_output);


    // --- 2. 反向遍历隐藏层 (从倒数第二层开始到第一层) ---
    for (int i = NUM_LAYERS - 3; i >= 0; i--) {
        Layer1* current_layer = &network[i];
        Layer1* next_layer = &network[i + 1];

        // 临时存储：用于存储加权后的 Delta 乘积，大小等于当前层的输出尺寸
        float* weighted_sum_of_deltas = (float*)malloc(current_layer->output_size * sizeof(float));
        if (weighted_sum_of_deltas == NULL) {
            fprintf(stderr, "Memory allocation error in train loop\n");
            exit(EXIT_FAILURE);
        }

        // 计算当前层的误差 (Delta): Delta_i = Delta_{i+1} * W_{i+1}^T 
        // next_layer->deltas: (1 x next_out), next_layer->weights: (next_in x next_out)
        // 结果 weighted_sum_of_deltas: (1 x current_out)
        matrix_multiply_transpose_b(next_layer->deltas, next_layer->weights, weighted_sum_of_deltas,
            1, next_layer->output_size, current_layer->output_size);

        // 元素级乘法 (Hadamard product) 应用 ReLU 激活函数的导数
        for (int j = 0; j < current_layer->output_size; j++) {
            float derivative = relu_derivative(current_layer->outputs[j]);
            current_layer->deltas[j] = weighted_sum_of_deltas[j] * derivative;
        }

        free(weighted_sum_of_deltas);
    }

    // --- 3. 更新所有权重和偏置 (从第一层到输出层) ---
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        Layer1* layer = &network[i];

        // 更新权重
        // 梯度 = Input^T * Delta (使用循环实现外积)
        for (int j = 0; j < layer->input_size; j++) { // Input neuron index
            for (int k = 0; k < layer->output_size; k++) { // Output neuron index
                float gradient = layer->inputs[j] * layer->deltas[k];
                // W[j][k] 的索引是 j * layer->output_size + k
                layer->weights[j * layer->output_size + k] -= LEARNING_RATE * gradient;
            }
        }

        // 更新偏置
        for (int k = 0; k < layer->output_size; k++) {
            layer->biases[k] -= LEARNING_RATE * layer->deltas[k];
        }
    }
}

void forward_pass1(const float* input, float* output)
{
    float* current_input = (float*)input;
    int current_input_size = network[0].input_size;

    for (int i = 0; i < NUM_LAYERS - 1; i++)
    {
        Layer1* layer = &network[i];

        // 1. 存储输入 (用于反向传播) 
        memcpy(layer->inputs, current_input, layer->input_size * sizeof(float));

        // 2. 线性运算: Z = Input * W + B
        // Input: (1 x InputSize), W: (InputSize x OutputSize), Z: (1 x OutputSize) 
        matrix_multiply(current_input, layer->weights, layer->outputs,
            1, layer->input_size, layer->output_size);

        // 3. 加上偏置并应用激活函数 (Outputs = Z + B) 
        for (int j = 0; j < layer->output_size; j++)
        {
            layer->outputs[j] += layer->biases[j];

            // 最后一层使用 Sigmoid，其他使用 ReLU 
            if (i == NUM_LAYERS - 2)
            {
                layer->outputs[j] = sigmoid(layer->outputs[j]);
            }
            else
            {
                layer->outputs[j] = relu(layer->outputs[j]);
            }
        }

        // 准备下一层输入 
        current_input = layer->outputs;
        current_input_size = layer->output_size;
    }

    // 复制最终输出
    memcpy(output, network[NUM_LAYERS - 2].outputs, network[NUM_LAYERS - 2].output_size * sizeof(float));
}// train 函数

int main() 
{
    // XOR 训练数据
    float training_inputs[4][2] = { {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0}, {1.0, 31.0} };
    float training_targets[4][1] = { {0.0}, {1.0}, {1.0}, {0.0} };

    initialize_network();
    printf("--- 10层网络初始化完成 ---\n");

    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < 4; i++) {
            train(training_inputs[i], training_targets[i]);

            // 计算损失 (用于显示)
            float output[1];
            forward_pass(training_inputs[i], output);
            float error = training_targets[i][0] - output[0];
            total_loss += error * error;
        }

        if (epoch % 2000 == 0) {
            printf("Epoch %d, Avg Loss: %f\n", epoch, total_loss / 4.0);
        }
    }

    printf("\n--- 训练结束，开始测试 ---\n");

        printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\r\n");
        // 测试
        for (int i = 0; i < 4; i++) 
        {
            float output[1];
            forward_pass(training_inputs[i], output);
            printf("Input: [%.1f, %.1f], Target: %.1f, Predicted: %.4f\n",
                training_inputs[i][0], training_inputs[i][1],
                training_targets[i][0], output[0]);
        }
     
    free_network();
    setup_dummy_network();

    // 调用 C++ 实现的保存函数
    save_weights_to_c_file_cpp("../Source/model_data_cpp.c");
    return 0;
}