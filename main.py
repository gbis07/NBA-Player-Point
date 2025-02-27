import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention

data = pd.read_csv('cleaned_data.csv')

original_player_names = data['PLAYER'].copy()

# Encode string variables
def encode_string(name):
     return ''.join(f'{ord(c):03d}' for c in name)

unique_teams = data['TEAM'].unique()
encoded_teams = {team: int(encode_string(team)) for team in unique_teams}
# print(encoded_teams)
data['TEAM'] = data['TEAM'].replace(encoded_teams)
# print(data.head())

unique_players = data['PLAYER'].unique()
encoded_players = {player: int(encode_string(player)) for player in unique_players}
data['PLAYER'] = data['PLAYER'].replace(encoded_players)
# print(data.head())

# Use start of season to denote
# i.e. 2003-2004 = 2003
def slice_season(season):
    return season[:4]
unique_seasons = data['Season'].unique()
sliced_seasons = {year: int(slice_season(year)) for year in unique_seasons}
data['Season'] = data['Season'].replace(sliced_seasons)
print(data.head())


targets = data.iloc[:, [7, 19, 22, 11, 12, 21, 23, 20]].values

# Separate Features from dataframe
feature_indices = [i for i in range(data.shape[1]) if i not in [0, 7, 11, 12, 19, 20, 21, 22, 23, 29]]
features = data.iloc[:, feature_indices].values

# Standardize features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()
features = scaler_X.fit_transform(features)
targets = scaler_y.fit_transform(targets)

# LSTM layers expect 3D input: (samples, timesteps, features)
# For demonstration, if each row is independent, we create a sequence length of 1
X = features.reshape((features.shape[0], 1, features.shape[1]))
y = targets

# Split into training and testing sets
X_train, X_test, y_train, y_test, player_train, player_test = train_test_split(X, y, original_player_names, test_size=0.2, random_state=42)

class PlayerStatsPredictor:
    def __init__(self, input_shape, lstm_units=128, num_heads=4, key_dim=32, dense_units=64, dropout_rate=0.2, output_dim=8):
        """
        Initialize the predictor with model parameters.
        :param input_shape: Tuple specifying the shape of the input (timesteps, features).
        :param lstm_units: Number of units in the LSTM layer.
        :param num_heads: Number of attention heads.
        :param key_dim: Dimensionality of the key vectors in attention.
        :param dense_units: Number of units in the dense layer.
        :param dropout_rate: Dropout rate after the dense layer.
        :param output_dim: Number of output metrics to predict.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
       
        inputs = Input(shape=self.input_shape)
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(inputs)
        attn_out = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)(lstm_out, lstm_out)
        pooled = GlobalAveragePooling1D()(attn_out)
        dense = Dense(self.dense_units, activation='relu')(pooled)
        dropout = Dropout(self.dropout_rate)(dense)
        outputs = Dense(self.output_dim)(dropout)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def summary(self):
        
        return self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
       
        if X_val is not None and y_val is not None:
            history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                     epochs=epochs, batch_size=batch_size)
        else:
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return history

    def predict(self, X):
    
        return self.model.predict(X)

    def evaluate(self, X, y):
        
        return self.model.evaluate(X, y)
    
# Define input shape
# X_train (num_samples, timesteps, num_features):
input_shape = (X_train.shape[1], X_train.shape[2])  # using timesteps and num_features from training data

predictor = PlayerStatsPredictor(input_shape=input_shape)
predictor.summary()
history = predictor.train(X_train, y_train, X_val=X_test, y_val=y_test, epochs=50, batch_size=32)

# --- Make Predictions on Test Set with Player Tracking ---
test_predictions = predictor.predict(X_test)
# Inverse-transform predictions to recover original scale.
original_test_predictions = scaler_y.inverse_transform(test_predictions)

# Labels for predicted metrics
labels = [
    "Expected points",
    "Expected rebounds",
    "Expected steals",
    "Expected 3P made",
    "Expected 3P attempted",
    "Expected turnovers",
    "Expected blocks",
    "Expected assists"
]

# Create a DataFrame to display predictions with player names
df_preds = pd.DataFrame(original_test_predictions, columns=labels)
df_preds['PLAYER'] = player_test.values  # Attach player names from the test split
cols = ['PLAYER'] + labels
df_preds = df_preds[cols]
print(df_preds.head())

# Evaluate on Test Data
loss = predictor.evaluate(X_test, y_test)
print("Test Loss:", loss)

df_preds.to_csv('predictions.csv', index=False)