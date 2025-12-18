from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_squared_log_obj(y_true: np.ndarray, y_pred: np.ndarray):
    '''
    Squared Log Error objective with Sigmoid transformation.
    
    y_pred: Raw logits (scores) from XGBoost
    y_true: The actual labels
    '''
    # 1. Apply Sigmoid to get predictions in range (0, 1)
    # y_pred comes in as raw logits (z), so we convert to p
    p = sigmoid(y_pred)
    
    # 2. Compute gradients w.r.t the transformed prediction (p)
    # These are the same formulas as your original templates, just using 'p' instead of 'predt'
    # Gradient_p: dL/dp
    grad_p = (np.log1p(p) - np.log1p(y_true)) / (p + 1)
    
    # Hessian_p: d2L/dp2
    hess_p = ((-np.log1p(p) + np.log1p(y_true) + 1) / np.power(p + 1, 2))
    
    # 3. Apply Chain Rule to get gradients w.r.t the raw logits (z)
    # derivative of sigmoid: p * (1 - p)
    sigmoid_deriv = p * (1.0 - p)
    
    # Gradient_z = Gradient_p * p'
    grad = grad_p * sigmoid_deriv
    
    # Hessian_z = Hessian_p * (p')^2 + Gradient_p * p''
    # where p'' = p(1-p)(1-2p)
    hess = hess_p * np.power(sigmoid_deriv, 2) + \
           grad_p * (sigmoid_deriv * (1.0 - 2.0 * p))
    
    return grad, hess

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTMAutoEncoder, self).__init__()
        self.seq_len = seq_len
        
        # Encoder: Compresses sequence to latent vector
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Decoder: Reconstructs sequence from latent vector
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # Encode: Get the final hidden state (latent representation)
        _, (hidden, _) = self.encoder(x)
        
        # Repeat hidden state for each time step in sequence to prime decoder
        # hidden shape: (1, batch, hidden_dim) -> permute -> (batch, 1, hidden_dim)
        latent = hidden.permute(1, 0, 2).repeat(1, self.seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(latent)
        
        # Map back to original feature space
        reconstructed = self.output_layer(decoded)
        return reconstructed

class LSTMDetector:
    def __init__(self, seq_len=30, hidden_dim=64, epochs=50, batch_size=64, lr=1e-3):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('mps')

    def _create_sequences(self, X):
        """
        Converts 2D data (N, F) into 3D sequences (N-seq_len+1, seq_len, F).
        """
        xs = []
        for i in range(len(X) - self.seq_len + 1):
            xs.append(X[i:(i + self.seq_len)])
        return np.array(xs)

    def fit(self, X, y=None):
        # 1. Scale Data
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Create Sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Convert to PyTorch tensors
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Initialize Model
        input_dim = X.shape[1]
        self.model = LSTMAutoEncoder(input_dim, self.hidden_dim, self.seq_len).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # 4. Training Loop
        self.model.train()
        print(f"Training LSTM on {self.device} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(x_batch)
                loss = criterion(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.6f}")
                
        return self

    def decision_function(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        # Run inference in batches
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq)), batch_size=self.batch_size, shuffle=False)
        mse_scores = []
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                reconstructed = self.model(x_batch)
                
                # Calculate MSE per sequence (batch, seq_len, features)
                # We average over sequence length and features
                loss = torch.mean((x_batch - reconstructed)**2, dim=[1, 2])
                mse_scores.extend(loss.cpu().numpy())
        
        mse_scores = np.array(mse_scores)
        
        # PAD the start so the output shape matches input X (for plotting)
        # The first 'seq_len-1' points don't have a full sequence to predict from.
        # We pad with the first valid score to keep plots aligned.
        padding = np.full(self.seq_len - 1, mse_scores[0])
        full_scores = np.concatenate([padding, mse_scores])
        
        # Convert to Health Score (0=Fault, 1=Healthy) using Exponential Decay
        return np.exp(-full_scores)

class AutoEncoderDetector:
    def __init__(self):
        # We separate the scaler from the model so we can scale inputs AND targets
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16, 32), 
            activation='tanh', 
            solver='adam', 
            random_state=42, 
            max_iter=200,
            early_stopping=True
        )

    def fit(self, X, y=None):
        # 1. Scale the data manually
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Train to reconstruct the SCALED data (Input=Scaled, Target=Scaled)
        self.model.fit(X_scaled, X_scaled) 
        return self

    def decision_function(self, X):
        # 1. Scale the test data using the same scaler
        X_scaled = self.scaler.transform(X)
        
        # 2. Predict (Reconstruct)
        reconstructed = self.model.predict(X_scaled)
        
        # 3. Calculate MSE in the SCALED space (Error will be small, e.g., 0.1 to 5.0)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
        
        # Return negative MSE (Higher is Normal, Lower is Anomaly)
        return np.exp(-mse)

def get_model(type):
    if type == 1:
        return AutoEncoderDetector()
    
    if type == 2:
        model = IsolationForest(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            contamination = 'auto',
            max_samples=10000
        )
        return model
    if type == 3:
        model = XGBRegressor(
            objective=sigmoid_squared_log_obj, 
            n_estimators=100, 
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            max_depth=4,
            
        )
        return model
    if type == 4: 
        # Sequence length 30 = 300 minutes (if 10 min intervals) history context
        return LSTMDetector(seq_len=30, hidden_dim=64, epochs=50)
