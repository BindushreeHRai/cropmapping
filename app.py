from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("puttur_crop_dataset12.csv")

# Encode categorical variables
label_encoders = {}
for column in ['Location (Village/Town)', 'Crop', 'Season', 'Soil Type', 'Rainfall Range (mm)']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and target
X = data[['Location (Village/Town)', 'Season', 'Soil Type', 'Min Temperature (°C)', 'Max Temperature (°C)']]
y = data['Crop']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        
        # Encode input data
        for column in ['Location (Village/Town)', 'Season', 'Soil Type', 'Rainfall Range (mm)']:
            if column in input_data:
                input_data[column] = label_encoders[column].transform(input_data[column])

        # Predict
        prediction = model.predict(input_data)
        predicted_crop = label_encoders['Crop'].inverse_transform(prediction)
        
        return jsonify({'predicted_crop': predicted_crop[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
