#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

// 一个最简单的单层神经网络（无隐藏层），用于学习逻辑与/或问题。
// 网络结构：输入层 -> 输出层（sigmoid激活）

static std::mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

double randd(double a = -1.0, double b = 1.0) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double dsigmoid(double y) { return y * (1.0 - y); }

using Vec = vector<double>;
using Mat = vector<Vec>;

Mat make_mat(int r, int c) {
    Mat m(r, Vec(c));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            m[i][j] = randd(-1.0, 1.0);
    return m;
}

Vec mat_mul_vec(const Mat& A, const Vec& x) {
    int r = (int)A.size();
    int c = (int)A[0].size();
    Vec y(r, 0.0);
    for (int i = 0; i < r; ++i) {
        double s = 0.0;
        for (int j = 0; j < c; ++j) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}

struct SingleLayerNN {
    int in_dim, out_dim;
    Mat W; // out_dim x in_dim
    Vec b; // out_dim

    SingleLayerNN(int in_, int out_) : in_dim(in_), out_dim(out_) {
        W = make_mat(out_dim, in_dim);
        b.assign(out_dim, 0.0);
    }

    Vec forward(const Vec& x) {
        Vec z = mat_mul_vec(W, x);
        for (int i = 0; i < out_dim; ++i) z[i] += b[i];
        for (int i = 0; i < out_dim; ++i) z[i] = sigmoid(z[i]);
        return z;
    }

    double train_sample(const Vec& x, const Vec& y_true, double lr) {
        Vec y_pred = forward(x);
        Vec delta(out_dim);
        for (int i = 0; i < out_dim; ++i)
            delta[i] = (y_pred[i] - y_true[i]) * dsigmoid(y_pred[i]);

        for (int i = 0; i < out_dim; ++i) {
            for (int j = 0; j < in_dim; ++j)
                W[i][j] -= lr * delta[i] * x[j];
            b[i] -= lr * delta[i];
        }

        double mse = 0.0;
        for (int i = 0; i < out_dim; ++i) {
            double e = y_pred[i] - y_true[i];
            mse += e * e;
        }
        return mse / out_dim;
    }
};

int main() {
    // 学习逻辑与(AND)运算
    vector<Vec> X = { {0,0}, {0,1}, {1,0}, {1,1} };
    vector<Vec> Y = { {0}, {0}, {0}, {1} };

	SingleLayerNN net(2, 1);// 2输入1输出 SingleLayerNN->

    const int epochs = 5000;
    const double lr = 0.5;

    for (int ep = 1; ep <= epochs; ++ep) 
    {
        double loss = 0.0;
        for (size_t i = 0; i < X.size(); ++i)
            loss += net.train_sample(X[i], Y[i], lr);
        if (ep % 500 == 0) cout << "Epoch " << ep << " loss=" << loss / X.size() << "\n";
    }

    cout << "训练完成，测试输出:\n";
    for (size_t i = 0; i < X.size(); ++i) 
    {
        Vec out = net.forward(X[i]);
        cout << "input(" << X[i][0] << "," << X[i][1] << ") -> " << out[0] << " (期望 " << Y[i][0] << ")\n";
    }

    return 0;
}

