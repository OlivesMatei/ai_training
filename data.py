import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

# Simulate data if not available
np.random.seed(42)
data_size = 1000
features = pd.DataFrame({
    'age': np.random.randint(20, 80, data_size),
    'gender': np.random.choice(['Male', 'Female'], data_size),
    'blood_pressure': np.random.randint(80, 180, data_size),
    'cholesterol': np.random.randint(150, 300, data_size),
    'glucose': np.random.randint(70, 200, data_size),
    # Add more features as necessary
})

# Target labels (simulate data)
features['diabetes'] = np.random.choice([0, 1], data_size)
features['heart_disease'] = np.random.choice([0, 1], data_size)
features['alzheimer'] = np.random.choice([0, 1], data_size)

# Encode categorical data
le = LabelEncoder()
features['gender'] = le.fit_transform(features['gender'])

# Split data into training and testing sets
X = features.drop(['diabetes', 'heart_disease', 'alzheimer'], axis=1)
y = features[['diabetes', 'heart_disease', 'alzheimer']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
