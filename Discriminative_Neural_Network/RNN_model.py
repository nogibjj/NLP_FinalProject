import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

nltk.download("punkt")

# simulation data
# Example data (replace with your actual data)
# Assume each email is converted to a sequence of integers
X = [[1, 2, 3, 4], [1, 4, 3, 5]]  # Email sequences
y = [0, 1]  # 0 for non-spam, 1 for spam

# Pad sequences for consistent input size
X_padded = pad_sequences(X, maxlen=100)  # Adjust 'maxlen' as needed
print(X_padded)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42
)

# build model
model = Sequential()
model.add(
    Embedding(input_dim=10000, output_dim=32)
)  # Adjust 'input_dim' as per your vocabulary size
model.add(SimpleRNN(32))  # 32 units in RNN layer
model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(
    X_train, y_train, epochs=10, batch_size=128
)  # Adjust epochs and batch_size as needed

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
