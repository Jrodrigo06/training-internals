#include <iostream>


double relu(double z) {
    return z > 0 ? z : 0;
}

double relu_derivative(double z) {
    return z > 0 ? 1.0 : 0.0;
}

struct DenseLayer {
    double w;
    double b;
    double cached_input;
    double cached_z;
    bool use_relu;
    
    DenseLayer(double init_w, double init_b, bool activate) {
        w = init_w;
        b = init_b;
        use_relu = activate;
    }
    
    double forward(double x) {
        cached_input = x;
        cached_z = w * cached_input + b;

        if (use_relu) {
            return relu(cached_z);
        }

        return cached_z;
    }
    
    void backward(double dL_d_output, double learning_rate) {
        double dL_dz = dL_d_output;
        if (use_relu) {
            dL_dz = dL_dz * relu_derivative(cached_z);
        }
        
        double dL_dw = dL_dz * cached_input;
        double dL_db = dL_dz;
        
        w = w - learning_rate * dL_dw;
        b = b - learning_rate * dL_db;
    }
};

int main() {

    double x_data[] = {1.0, 2.0, 3.0, 4.0};
    double y_data[] = {3.0, 5.0, 7.0, 9.0};

    bool relu = true;

    DenseLayer layer(1.0, -1.0, relu);

    double learning_rate = 0.01;
    int epochs = 1000;
    
    for(int j = 0; j < epochs; j++) {
       
        double total_loss = 0.0;

        for (int k = 0; k < (sizeof(x_data) / sizeof (x_data[0])); k++) {
            double x = x_data[k];
            double y = y_data[k];

            double y_pred = layer.forward(x);

            double loss = (y - y_pred) * (y - y_pred);
            total_loss += loss;

            double dl_dy_pred = -2 * (y - y_pred);

            layer.backward(dl_dy_pred, learning_rate);
        }

        total_loss = total_loss / 4;

        
        if (j % 100 == 0) {
            std::cout << "ReLU: " << relu << " Epoch " << j << ": Avg Loss = " << total_loss 
          << ", w = " << layer.w << ", b = " << layer.b << std::endl;
        }
    }
}