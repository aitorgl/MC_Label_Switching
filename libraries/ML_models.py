#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:58:28 2025

@author: fran
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ---------------------------------------------
# PyTorch implementation of a custom MLP
# ---------------------------------------------
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', 
                 alpha=0.0, beta=0.0, alpha_tr=0.0001, batch_size='auto', 
                 learning_rate_init=0.001, max_iter=200, solver='adam'):
        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.beta = beta
        self.alpha_tr = alpha_tr
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.solver = solver

        # Build layers
        layer_sizes = [input_dim] + list(hidden_layer_sizes) + [1]
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])

        # Initialize weights
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Optimizer
        if solver == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha_tr)
        elif solver == 'lbfgs':
            self.optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate_init)
        else:
            raise ValueError("Only 'adam' and 'lbfgs' are supported solvers.")

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(x)
        
        # Output layer with custom transformation
        z = self.layers[-1](x)
        return torch.where(
            z < 0,
            torch.tanh(z) * (1 - 2 * self.alpha),
            torch.tanh(z) * (1 - 2 * self.beta),
        )

    def fit(self, X, y):
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        batch_size = len(X) if self.batch_size == 'auto' else self.batch_size
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
        criterion = nn.MSELoss()  # Change from BCELoss to MSELoss
    
        for epoch in range(self.max_iter):
            for batch_X, batch_y in dataloader:
                def closure():
                    self.optimizer.zero_grad()
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs, batch_y)  # Compute MSE loss
                    loss.backward()
                    return loss
    
                if self.solver == 'adam':
                    loss = closure()
                    self.optimizer.step()
                elif self.solver == 'lbfgs':
                    self.optimizer.step(closure)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.forward(X)
        return (y_pred > 0.0).int().numpy()

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.forward(X)
        return torch.cat([1 - y_pred, y_pred], dim=1).numpy()
    

# ---------------------------------------------
# PyTorch implementation of a custom LogReg
# ---------------------------------------------
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, alpha, beta, learning_rate=0.01, num_epochs=100):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def forward(self, x):
        z = self.linear(x)
        return torch.where(
            z < 0,
            torch.tanh(z) * (1 - 2 * self.alpha),
            torch.tanh(z) * (1 - 2 * self.beta)
        )

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
