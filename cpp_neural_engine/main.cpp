#include <iostream>

int main() {
    double x = 2;
    double y = 5;

    double w = 1;
    double b = 1;

    double learning_rate = 0.01;
    int epochs = 1000;
    
    for(int j = 0; j < epochs; j++) {
       
        double y_pred = w * x + b;
       
        double loss = (y - y_pred) * (y - y_pred);

        double dl_dw = -2 * (y - y_pred) * x;
        double dl_db = -2 * (y - y_pred);

        w = w - learning_rate * dl_dw;
        b = b - learning_rate * dl_db;

        if (j % 100 == 0) {
            std::cout << "Epoch " << j << ": Loss = " << loss 
          << ", w = " << w << ", b = " << b << std::endl;
        }
    }
}