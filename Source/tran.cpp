#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE  4
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 3
#define LEARNING_RATE 0.01
#define EPOCHS 2000

// ------------------ 工具函数 ------------------
float randf() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

void softmax(float* z, float* out, int n) {
    float max = z[0];
    for (int i = 1; i < n; i++)
        if (z[i] > max) max = z[i];

    float sum = 0;
    for (int i = 0; i < n; i++) {
        out[i] = expf(z[i] - max);
        sum += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

// ------------------ 网络权重 ------------------
float W1[HIDDEN_SIZE][INPUT_SIZE];
float b1[HIDDEN_SIZE];
float W2[OUTPUT_SIZE][HIDDEN_SIZE];
float b2[OUTPUT_SIZE];

// ------------------ 初始化权重 ------------------
void init_weights() {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++)
            W1[i][j] = randf() * 0.1;
        b1[i] = 0;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W2[i][j] = randf() * 0.1;
        b2[i] = 0;
    }
}

// ------------------ 前向传播 ------------------
void forward(float* x,
    float* z1, float* a1,
    float* z2, float* a2)
{
    // 第一层
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        z1[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            z1[i] += W1[i][j] * x[j];
        a1[i] = (z1[i] > 0 ? z1[i] : 0);
    }

    // 输出层
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        z2[i] = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            z2[i] += W2[i][j] * a1[j];
    }

    softmax(z2, a2, OUTPUT_SIZE);
}

// ------------------ 反向传播 ------------------
void backward(float* x, float* a1, float* a2, int label,
    float* dW1, float* db1, float* dW2, float* db2)
{
    float dz2[OUTPUT_SIZE];

    // softmax + cross entropy
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        dz2[i] = a2[i] - (i == label ? 1.0f : 0.0f);
    }

    // dW2, db2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            dW2[i * HIDDEN_SIZE + j] = dz2[i] * a1[j];
        db2[i] = dz2[i];
    }

    // 反传到隐藏层
    float da1[HIDDEN_SIZE] = { 0 };
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            da1[j] += W2[i][j] * dz2[i];
    }

    // ReLU 导数
    float dz1[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++)
        dz1[i] = (a1[i] > 0 ? da1[i] : 0);

    // dW1, db1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++)
            dW1[i * INPUT_SIZE + j] = dz1[i] * x[j];
        db1[i] = dz1[i];
    }
}

// ------------------ 参数更新 ------------------
void update(float* dW1, float* db1, float* dW2, float* db2)
{
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++)
            W1[i][j] -= LEARNING_RATE * dW1[i * INPUT_SIZE + j];
        b1[i] -= LEARNING_RATE * db1[i];
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            W2[i][j] -= LEARNING_RATE * dW2[i * HIDDEN_SIZE + j];
        b2[i] -= LEARNING_RATE * db2[i];
    }
}

// ------------------ 示例训练数据 ------------------
float X_train[6][INPUT_SIZE] = {
    {1,0,0,0}, {1,1,0,0}, {0,1,0,0},   // 类 0
    {0,0,1,1}, {0,0,1,0}, {0,0,0,1},   // 类 1 or 2
};

int Y_train[6] = { 0,0,0, 1,1,2 };

// ------------------ 训练函数 ------------------
void train() {
    float z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];
    float z2[OUTPUT_SIZE], a2[OUTPUT_SIZE];

    float dW1[HIDDEN_SIZE * INPUT_SIZE];
    float dW2[OUTPUT_SIZE * HIDDEN_SIZE];
    float db1[HIDDEN_SIZE];
    float db2[OUTPUT_SIZE];

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 6; i++) {

            forward(X_train[i], z1, a1, z2, a2);

            backward(X_train[i], a1, a2, Y_train[i],
                dW1, db1, dW2, db2);

            update(dW1, db1, dW2, db2);
        }
    }
}

// ------------------ 测试函数 ------------------
void test() {
    float z1[HIDDEN_SIZE], a1[HIDDEN_SIZE];
    float z2[OUTPUT_SIZE], a2[OUTPUT_SIZE];

    int correct = 0;

    for (int i = 0; i < 6; i++) {
        forward(X_train[i], z1, a1, z2, a2);

        int pred = 0;
        float maxp = a2[0];
        for (int k = 1; k < OUTPUT_SIZE; k++) {
            if (a2[k] > maxp) {
                maxp = a2[k];
                pred = k;
            }
        }

        printf("Sample %d: pred = %d (prob = %.3f)\n", i, pred, maxp);

        if (pred == Y_train[i]) correct++;
    }

    printf("\nAccuracy = %.2f%%\n",
        correct * 100.0f / 6);
}

// ------------------ 主函数 ------------------
int main() {
    srand(time(NULL));
    init_weights();

    printf("Training...\n");
    train();

    printf("Testing...\n");
    test();

    return 0;
}
