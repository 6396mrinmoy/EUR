import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

# Load Dataset (Assuming a CSV file with relevant columns)
df = pd.read_csv("eur_dataset.csv")  # Replace with actual dataset file

# Data Preprocessing
# Drop irrelevant columns (adjust based on correlation analysis)
df.drop(columns=['Pressure Gradient'], inplace=True, errors='ignore')

# Splitting Features and Target
X = df.drop(columns=['EUR'])  # Assuming 'EUR' is the target variable
y = df['EUR']

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Random Forest Model
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

# Support Vector Machine Model
svm = SVR(kernel='poly', degree=4)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM R2 Score:", r2_score(y_test, y_pred_svm))

# Decision Tree Model
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree R2 Score:", r2_score(y_test, y_pred_dt))

# Neural Network Model
model = Sequential([
    Normalization(input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(12, activation='relu'),
    Dense(1)  # Output Layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))
y_pred_nn = model.predict(X_test).flatten()
print("Neural Network R2 Score:", r2_score(y_test, y_pred_nn))

# Plot Actual vs Predicted
def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual EUR")
    plt.ylabel("Predicted EUR")
    plt.title(title)
    plt.show()

plot_results(y_test, y_pred_rf, "Random Forest")
plot_results(y_test, y_pred_svm, "SVM")
plot_results(y_test, y_pred_dt, "Decision Tree")
plot_results(y_test, y_pred_nn, "Neural Network")
