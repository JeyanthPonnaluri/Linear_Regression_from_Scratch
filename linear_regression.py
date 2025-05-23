import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def loss_function(m,b,data):

    for i in range(len(data)): 
        x = data[i][0]
        y = data[i][1]
        y_pred = m*x + b
        loss = (y - y_pred)**2
        total_loss+=loss

    return total_loss/float(len(data))

def gradient_descent(m, b, data, l, epochs):
    m_gradient=0
    b_gradient=0
    n=len(data)

    for i in range(n):
        x = data[i][0]
        y = data[i][1]
        y_pred = m*x + b
        m_gradient += -(2/n) * x * (y - y_pred)
        b_gradient += -(2/n) * (y - y_pred)

    m -= l * m_gradient
    b -= l * b_gradient

    return m, b

def linear_regression(data, learning_rate=0.01, epochs=1000):
    m = 0
    b = 0

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, learning_rate, epochs)
        if i % 100 == 0:
            print(f"Epoch {i}: Loss = {loss_function(m, b, data)}")

    return m, b

def plot_regression_line(m, b, data):
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data points')
    plt.plot(data[:, 0], m * data[:, 0] + b, color='red', label='Regression line')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Generate random data
    x=np.linspace(0, 10, 100)
    y=2*x+3+np.random.randn(100)
    data = pd.DataFrame({'x': x, 'y': y})
    data = data.values
    # Convert data to numpy array
    data = np.array(data)
    
    
    # Train the model
    m, b = linear_regression(data, learning_rate=0.01, epochs=1000)
    
    # Print the final parameters
    print(f"Final parameters: m = {m}, b = {b}")
    
    # Plot the regression line
    plot_regression_line(m, b, data)