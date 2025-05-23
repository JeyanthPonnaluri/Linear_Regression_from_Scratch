# Linear Regression from Scratch

---

## Introduction

Linear Regression is a fundamental algorithm in machine learning and statistics. It models the relationship between a dependent variable \( y \) and one or more independent variables \( X \) by fitting a straight line.

---

## Mathematical Foundation

### 1. Hypothesis Function

The linear model assumes a linear relationship:

\[
\hat{y} = m x + b
\]

Where:  
- \( \hat{y} \): predicted value  
- \( m \): slope of the line  
- \( b \): y-intercept

---

### 2. Loss Function

To measure how well the model fits the data, we use the **Mean Squared Error (MSE)** loss:

\[
\text{Loss} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

Where:  
- \( n \): number of data points  
- \( y_i \): actual value  
- \( \hat{y}_i \): predicted value

---

### 3. Gradient Descent

To minimize the loss, we use gradient descent — an iterative optimization algorithm that updates \( m \) and \( b \) in the direction that reduces loss:

Partial derivatives of loss w.r.t parameters:

\[
\frac{\partial \text{Loss}}{\partial m} = -\frac{2}{n} \sum_{i=1}^n x_i (y_i - \hat{y}_i)
\]

\[
\frac{\partial \text{Loss}}{\partial b} = -\frac{2}{n} \sum_{i=1}^n (y_i - \hat{y}_i)
\]

Update rules:

\[
m = m - \alpha \cdot \frac{\partial \text{Loss}}{\partial m}
\]

\[
b = b - \alpha \cdot \frac{\partial \text{Loss}}{\partial b}
\]

Where:  
- \( \alpha \) is the learning rate (step size).

---

## Code Approach Overview

1. **Initialize parameters** \( m \) and \( b \) to zero.  
2. For a given number of **epochs** (iterations):  
   - Calculate predicted values \( \hat{y} \) using the current parameters.  
   - Compute the loss using the Mean Squared Error function.  
   - Compute gradients of the loss with respect to \( m \) and \( b \).  
   - Update parameters \( m \) and \( b \) by moving against the gradients, scaled by the learning rate \( \alpha \).  
3. After training, the parameters \( m \) and \( b \) should approximate the underlying linear relationship in the data.  
4. Plot the data points and the fitted regression line to visualize the result.

---

## Usage

- Generate synthetic linear data with noise.  
- Train the model using gradient descent.  
- Monitor the loss every 100 epochs.  
- Visualize the regression line against the data.

---

This project provides a clear understanding of the mechanics behind linear regression and gradient descent without relying on external ML libraries.
