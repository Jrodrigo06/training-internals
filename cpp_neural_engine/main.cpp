#include <iostream>

int main() {

    double x_data[] = {1.0, 2.0, 3.0, 4.0};
    double y_data[] = {3.0, 5.0, 7.0, 9.0};

    double w = 1;
    double b = 1;

    double learning_rate = 0.01;
    int epochs = 1000;
    
    for(int j = 0; j < epochs; j++) {
       
        double total_loss = 0.0;

        for (int k = 0; k < (sizeof(x_data) / sizeof (x_data[0])); k++) {
            double x = x_data[k];
            double y = y_data[k];

            double y_pred = w * x + b;

            double loss = (y - y_pred) * (y - y_pred);
            total_loss += loss;

            double dl_dw = -2 * (y - y_pred) * x;
            double dl_db = -2 * (y - y_pred);

            w = w - learning_rate * dl_dw;
            b = b - learning_rate * dl_db;
        }

        total_loss = total_loss / 4;

        

        if (j % 100 == 0) {
            std::cout << "Epoch " << j << ": Avg Loss = " << total_loss 
          << ", w = " << w << ", b = " << b << std::endl;
        }
    }
}