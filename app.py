#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CKD Mortality Risk Calculator - Flask Version
Predicts 20-year all-cause and cardiovascular mortality risk
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gzip
import shutil

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SHAP_BG_ALL_CAUSE_PATH = os.path.join(BASE_DIR, "shap_background_all_cause.csv")
SHAP_BG_CARDIO_PATH = os.path.join(BASE_DIR, "shap_background_cardio.csv")

# Time horizon: 240 months (20 years)
TIME_HORIZON = 240.0

# Helper function to load compressed models
def load_model(model_path):
    """Load model, decompress if needed"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    
    # Check for compressed version
    compressed_path = model_path + ".gz"
    if os.path.exists(compressed_path):
        print(f"Decompressing {os.path.basename(compressed_path)}...")
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Decompression complete!")
        return joblib.load(model_path)
    
    raise FileNotFoundError(f"Model not found: {model_path}")

# Load models and data
print("Loading models and SHAP background data...")
scaler = joblib.load(SCALER_PATH)
model_all_cause = load_model(os.path.join(MODEL_DIR, "CI_all_cause_death_GradientBoostingSurvival.pkl"))
model_cardiovascular = load_model(os.path.join(MODEL_DIR, "CI_cardiovascular_death_RandomSurvivalForest.pkl"))
shap_bg_all_cause = pd.read_csv(SHAP_BG_ALL_CAUSE_PATH)
shap_bg_cardio = pd.read_csv(SHAP_BG_CARDIO_PATH)

# Feature definitions
all_cause_features = ['WBC', 'Dyslipidemia', 'DBP', 'Creatinine', 'Glucose', 'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI']
cardiovascular_features = ['WBC', 'DBP', 'UricAcid', 'SBP', 'BUN', 'Age', 'CI']
categorical_features = ['Gender', 'smoking', 'HighCholesterol', 'Dyslipidemia']

print("Application loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'all_cause')
        
        # Select features and model based on type
        if model_type == 'all_cause':
            features = all_cause_features
            model = model_all_cause
            shap_bg = shap_bg_all_cause
        else:
            features = cardiovascular_features
            model = model_cardiovascular
            shap_bg = shap_bg_cardio
        
        # Extract input values
        input_dict = {}
        for feature in features:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing value for {feature}'}), 400
            input_dict[feature] = float(value)
        
        # Create input dataframe
        input_df = pd.DataFrame([input_dict])
        
        # Standardize features (excluding categorical)
        features_to_scale = [f for f in features if f not in categorical_features]
        if features_to_scale:
            input_df_scaled = input_df.copy()
            input_df_scaled[features_to_scale] = scaler.transform(input_df[features_to_scale])
        else:
            input_df_scaled = input_df
        
        # Get survival function
        surv_func = model.predict_survival_function(input_df_scaled)
        
        # Find probability at TIME_HORIZON
        if hasattr(surv_func[0], 'x') and hasattr(surv_func[0], 'y'):
            times = surv_func[0].x
            probabilities = surv_func[0].y
        else:
            times = model.unique_times_
            probabilities = surv_func[0]
        
        # Interpolate probability at TIME_HORIZON
        if TIME_HORIZON <= times[0]:
            survival_prob = probabilities[0]
        elif TIME_HORIZON >= times[-1]:
            survival_prob = probabilities[-1]
        else:
            survival_prob = np.interp(TIME_HORIZON, times, probabilities)
        
        mortality_risk = (1 - survival_prob) * 100
        
        # Generate SHAP explanation
        explainer = shap.TreeExplainer(model, shap_bg[features])
        shap_values = explainer.shap_values(input_df_scaled[features])
        
        # Create SHAP waterfall plot
        plt.figure(figsize=(10, 6))
        if len(shap_values.shape) == 3:
            shap_values_plot = shap_values[0, :, 0]
        else:
            shap_values_plot = shap_values[0]
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_plot,
                base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
                data=input_df[features].values[0],
                feature_names=features
            ),
            show=False
        )
        
        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'mortality_risk': round(mortality_risk, 2),
            'survival_probability': round(survival_prob * 100, 2),
            'shap_plot': plot_base64
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
