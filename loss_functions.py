def mse_loss_derivative(y_real, y_predicted):
    return 2 * (y_predicted - y_real) / y_real.size
