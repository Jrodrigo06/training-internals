#include <iostream>

struct DenseLayer {
    double w;
    double b;
    double cached_input;
    
    DenseLayer(double init_w, double init_b) {
        w = init_w;
        b = init_b;
    }
    
    double forward(double x) {
        cached_input = x;
        return w * cached_input + b;
    }
    
    void backward(double grad_from_loss, double learning_rate) {
        w = w - learning_rate * (cached_input * grad_from_loss);
        b = b - learning_rate * grad_from_loss;
    }
};

int main() {

    double x_data[] = {1.0, 2.0, 3.0, 4.0};
    double y_data[] = {3.0, 5.0, 7.0, 9.0};

    DenseLayer layer(1.0, 1.0);

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
            std::cout << "Epoch " << j << ": Avg Loss = " << total_loss 
          << ", w = " << layer.w << ", b = " << layer.b << std::endl;
        }
    }
}