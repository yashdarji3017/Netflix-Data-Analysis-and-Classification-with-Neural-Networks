import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r'D:\Python\Dataset Programs\netflix_titles.csv')

# Display initial information
print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())

# Data preprocessing
content_type_count = df['type'].value_counts()
print("Content type distribution:\n", content_type_count)

# Drop unnecessary columns
df = df.drop(columns=['director', 'cast'])

# Fill missing values
df['country'].fillna(df['country'].mode()[0], inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)
df['date_added'].fillna('Unknown', inplace=True)

# Drop rows with missing values in 'rating' and 'duration'
df = df.dropna(subset=['rating', 'duration'])
print(df.isnull().sum())

# Prepare features and target
X = df.drop(['show_id', 'type'], axis=1)
y = df['type']

# Encode categorical features and target
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)

# Scale features using StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE (resampling):")
print(pd.Series(y_train).value_counts())

# Define neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=20, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss}, Test Accuracy: {accuracy}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()

# Predictions and evaluation
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

cm = confusion_matrix(y_test, binary_predictions)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_test, binary_predictions))
