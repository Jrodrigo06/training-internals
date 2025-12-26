#include <iostream>
#include <cmath>
#include <fstream>

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

    double last_dL_dw;
    double last_dL_db;
    
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
    
    double backward(double dL_d_output) {
        double dL_dz = dL_d_output;
        if (use_relu) {
            dL_dz = dL_dz * relu_derivative(cached_z);
        }
        
        last_dL_dw = dL_dz * cached_input;
        last_dL_db = dL_dz;
        double dL_dx = dL_dz * w;
        return dL_dx;
    }
    
    void update_params(double avg_dw, double avg_db, double learning_rate) {
        w = w - learning_rate * avg_dw;
        b = b - learning_rate * avg_db;
    }

    double get_grad_norm() {
        return std::sqrt(last_dL_db * last_dL_db + last_dL_dw * last_dL_dw);
    }
};

int main() {

    double x_data[] = {1.0, 2.0, 3.0, 4.0};
    double y_data[] = {3.0, 5.0, 7.0, 9.0};

    bool relu = true;

    DenseLayer layer1 = DenseLayer(1.0, 1.0, relu);
    DenseLayer layer2 = DenseLayer(1.0, 1.0, false);

    double learning_rate = 0.01;
    int epochs = 1000;
    
    std::ofstream logfile("../data/training_log_dead_neuron.csv");
    logfile << "epoch,loss,w1,b1,w2,b2,grad_norm\n";

    int N = sizeof(x_data) / sizeof(x_data[0]);
    
    for(int j = 0; j < epochs; j++) {
       
        double total_loss = 0.0;
        double sum_dW1 = 0.0;
        double sum_db1 = 0.0;
        double sum_dW2 = 0.0;
        double sum_db2 = 0.0;

        for (int k = 0; k < N; k++) {
            double x = x_data[k];
            double y = y_data[k];

            double h = layer1.forward(x);
            double y_pred = layer2.forward(h);


            double loss = (y - y_pred) * (y - y_pred);
            total_loss += loss;

            double dl_dy_pred = 2.0 * (y_pred - y);

            double dl_dh = layer2.backward(dl_dy_pred);

            layer1.backward(dl_dh);
            
            sum_dW1 += layer1.last_dL_dw;
            sum_db1 += layer1.last_dL_db;
            sum_dW2 += layer2.last_dL_dw;
            sum_db2 += layer2.last_dL_db;
        }

        double avg_dW1 = sum_dW1 / N;
        double avg_db1 = sum_db1 / N;
        double avg_dW2 = sum_dW2 / N;
        double avg_db2 = sum_db2 / N;
        
        layer1.update_params(avg_dW1, avg_db1, learning_rate);
        layer2.update_params(avg_dW2, avg_db2, learning_rate);
        
        double grad_norm = std::sqrt(avg_dW1 * avg_dW1 + avg_db1 * avg_db1 + avg_dW2 * avg_dW2 + avg_db2 * avg_db2);
        
        double avg_loss = total_loss / N;

        logfile << j << "," << avg_loss << "," 
            << layer1.w << "," << layer1.b << "," 
            << layer2.w << "," << layer2.b << "," 
            << grad_norm << "\n";

        
        if (j % 100 == 0) {
            std::cout << "Epoch " << j << ": Loss = " << avg_loss 
                  << ", w1 = " << layer1.w << ", b1 = " << layer1.b 
                  << ", w2 = " << layer2.w << ", b2 = " << layer2.b 
                  << ", grad_norm = " << grad_norm << std::endl;
        }
    }

    logfile.close();
    std::cout << "Training log saved to training_log.csv" << std::endl;
}