#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-based Risk Calculator using Flask
Supports both All-cause and Cardiovascular Mortality Prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import shap
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set font to Times New Roman globally for matplotlib
# Added try-except for deployment environments where the font might be missing
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
except Exception as e:
    print(f"Warning: Could not set font to Times New Roman. Using default. Error: {e}")

app = Flask(__name__)

# Configuration
# Use relative paths for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SHAP_BG_ALL_CAUSE_PATH = os.path.join(BASE_DIR, "shap_background_all_cause.csv")
SHAP_BG_CARDIO_PATH = os.path.join(BASE_DIR, "shap_background_cardio.csv")

# Load models and scaler
print("Loading models and scaler...")
try:
    scaler = joblib.load(SCALER_PATH)
    model_all_cause = joblib.load(os.path.join(MODEL_DIR, "CI_all_cause_death_GradientBoostingSurvival.pkl"))
    model_cardiovascular = joblib.load(os.path.join(MODEL_DIR, "CI_cardiovascular_death_RandomSurvivalForest.pkl"))
    
    # Load SHAP background data (pre-computed cluster centers)
    shap_bg_all_cause = pd.read_csv(SHAP_BG_ALL_CAUSE_PATH)
    shap_bg_cardio = pd.read_csv(SHAP_BG_CARDIO_PATH)
    print(f"Loaded SHAP background data: all-cause={shap_bg_all_cause.shape}, cardio={shap_bg_cardio.shape}")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please ensure all required files are present")
    # Create dummy objects to prevent immediate crash, but app won't work correctly
    scaler = None
    model_all_cause = None
    model_cardiovascular = None
    shap_bg_all_cause = pd.DataFrame()
    shap_bg_cardio = pd.DataFrame()

# Prediction Time Horizon (Months)
# Setting to 240 months (20 years) for better discrimination and clinical relevance
TIME_HORIZON = 240.0 

# Get all feature names that scaler expects
if scaler and hasattr(scaler, 'feature_names_in_'):
    scaler_features = list(scaler.feature_names_in_)
else:
    scaler_features = []

# Feature definitions
all_cause_features = [
    'WBC', 'Dyslipidemia', 'DBP', 'Creatinine', 'Glucose', 
    'Gender', 'TG', 'SBP', 'Age', 'MCV', 'smoking', 'Platelet', 'CI'
]

cardiovascular_features = [
    'WBC', 'DBP', 'UricAcid', 'SBP', 'BUN', 'Age', 'CI'
]

# Define categorical features (not standardized)
categorical_features = ['Gender', 'smoking', 'HighCholesterol', 'Dyslipidemia']

# Feature name mapping (English display names)
feature_mapping = {
    'WBC': 'White Blood Cell (10^9/L)',
    'Dyslipidemia': 'Dyslipidemia',
    'DBP': 'Diastolic Blood Pressure (mmHg)',
    'Creatinine': 'Serum Creatinine (mg/dL)',
    'Glucose': 'Blood Glucose (mg/dL)',
    'Gender': 'Sex',
    'TG': 'Triglycerides (mg/dL)',
    'SBP': 'Systolic Blood Pressure (mmHg)',
    'Age': 'Age (years)',
    'MCV': 'Mean Corpuscular Volume (fL)',
    'smoking': 'Smoking Status',
    'Platelet': 'Platelet Count (10^9/L)',
    'CI': 'Conicity Index',
    'UricAcid': 'Uric Acid (mg/dL)',
    'BUN': 'Blood Urea Nitrogen (mg/dL)'
}

# Feature ranges and units
feature_info = {
    'WBC': {'min': 1.0, 'max': 30.0, 'step': 0.1, 'unit': '10^9/L', 'type': 'number'},
    'Dyslipidemia': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 1, 'label': 'Yes (have dyslipidemia)'}, 
            {'value': 2, 'label': 'No (no dyslipidemia)'}
        ]
    },
    'DBP': {'min': 40, 'max': 120, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Creatinine': {'min': 0.1, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'Glucose': {'min': 50, 'max': 500, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'Gender': {
        'min': 1, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 1, 'label': 'Male'}, 
            {'value': 2, 'label': 'Female'}
        ]
    },
    'TG': {'min': 20, 'max': 1000, 'step': 1, 'unit': 'mg/dL', 'type': 'number'},
    'SBP': {'min': 80, 'max': 220, 'step': 1, 'unit': 'mmHg', 'type': 'number'},
    'Age': {'min': 18, 'max': 100, 'step': 1, 'unit': 'years', 'type': 'number'},
    'MCV': {'min': 60, 'max': 120, 'step': 0.1, 'unit': 'fL', 'type': 'number'},
    'smoking': {
        'min': 0, 'max': 2, 'step': 1, 'unit': '', 'type': 'select', 
        'options': [
            {'value': 0, 'label': 'Never'}, 
            {'value': 1, 'label': 'Former'}, 
            {'value': 2, 'label': 'Current'}
        ]
    },
    'Platelet': {'min': 10, 'max': 1000, 'step': 1, 'unit': '10^9/L', 'type': 'number'},
    'CI': {'min': 1.0, 'max': 1.8, 'step': 0.01, 'unit': '', 'type': 'number'},
    'UricAcid': {'min': 1.0, 'max': 15.0, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'},
    'BUN': {'min': 1.0, 'max': 150, 'step': 0.1, 'unit': 'mg/dL', 'type': 'number'}
}

# Initialize SHAP explainers
print("Initializing SHAP explainers (this may take a moment)...")
# Use ALL training data (approximately 7000 samples) for SHAP background
if not df_train.empty:
    X_all_cause = df_train[all_cause_features]
    X_cardio = df_train[cardiovascular_features]

    print(f"Using {len(X_all_cause)} training samples for SHAP background data...")
else:pre-computed background data (100 cluster centers)
if not shap_bg_all_cause.empty and not shap_bg_cardio.empty:
    X_all_cause = shap_bg_all_cause
    X_cardio = shap_bg_cardio
    print(f"Using pre-computed SHAP background: all-cause={len(X_all_cause)}, cardio={len(X_cardio)} samples")
else:
    print("Warning: SHAP backgroundne:
        return np.array([0.0])
        
    if hasattr(model_all_cause, "predict_survival_function"):
        surv_funcs = model_all_cause.predict_survival_function(data)
        death_probs = []
        for surv_func in surv_funcs:
            # Get survival probability at TIME_HORIZON
            surv_prob = surv_func(TIME_HORIZON)
            death_prob = 1.0 - surv_prob
            death_probs.append(death_prob)
        return np.array(death_probs)
    else:
        # Fallback for models without survival function (e.g. classifiers)
        risk_scores = model_all_cause.predict(data)
        return 1.0 / (1.0 + np.exp(-risk_scores))

def predict_cardio(data):
    """Predict cardiovascular mortality probability at 20 years"""
    if model_cardiovascular is None:
        return np.array([0.0])

    if hasattr(model_cardiovascular, "predict_survival_function"):
        surv_funcs = model_cardiovascular.predict_survival_function(data)
        death_probs = []
        for surv_func in surv_funcs:
            # Get survival probability at TIME_HORIZON
            surv_prob = surv_func(TIME_HORIZON)
            death_prob = 1.0 - surv_prob
            death_probs.append(death_prob)
        return np.array(death_probs)
    elif hasattr(model_cardiovascular, "predict_proba"):
        probs = model_cardiovascular.predict_proba(data)
        return probs[:, 1] if probs.shape[1] == 2 else probs
    return model_cardiovascular.predict(data)

# Calculate Risk Thresholds based on Youden Index (20-Year)
# Derived from calculate_youden_index.py results
print("Setting risk thresholds based on 20-Year Youden Index...")
thresholds = {
    'all_cause': {
        'low': 0.12,   # 12.0% (Half of optimal cutoff 24.0%)
        'high': 0.24   # 24.0% (Optimal Youden cutoff)
    },
    'cardio': {
        'low': 0.06,   # 6.0% (Half of optimal cutoff 11.75%)
        'high': 0.12   # 12.0% (Rounded optimal Youden cutoff)
    }
}
print(f"Risk Thresholds (20-Year):")
print(f"  All-cause: Low < {thresholds['all_cause']['low']:.1%} | Medium | High > {thresholds['all_cause']['high']:.1%}")
print(f"  Cardio:    Low < {thresholds['cardio']['low']:.1%} | Medium | High > {thresholds['cardio']['high']:.1%}")

# Create SHAP explainers
# Cluster all training data into representative points for computational efficiency
# Use pre-computed cluster centers directly (no need to re-cluster)
if X_all_cause is not None and X_cardio is not None:
    print("Initializing SHAP explainers with pre-computed background data...")
    try:
        explainer_all_cause = shap.KernelExplainer(predict_all_cause, X_all_cause)
        explainer_cardio = shap.KernelExplainer(predict_cardio, X_cardio)
        print(f"SHAP explainers initialized successfully with {len(X_all_cause)} background sample
        print(f"Error initializing SHAP explainers: {e}")
        explainer_all_cause = None
        explainer_cardio = None
else:
    explainer_all_cause = None
    explainer_cardio = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         all_cause_features=all_cause_features,
                         cardiovascular_features=cardiovascular_features,
                         feature_mapping=feature_mapping,
                         feature_info=feature_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.json
        model_type = data.get('model_type', 'all_cause')
        
        # Select model and features
        if model_type == 'all_cause':
            features = all_cause_features
            explainer = explainer_all_cause
            predict_fn = predict_all_cause
        else:
            features = cardiovascular_features
            explainer = explainer_cardio
            predict_fn = predict_cardio
        
        if explainer is None:
             return jsonify({
                'success': False,
                'error': "Model or Explainer not initialized. Please check server logs."
            })

        # Extract input values
        input_values = []
        for feat in features:
            value = float(data.get(feat, 0))
            input_values.append(value)
        
        # Create DataFrame with original values
        input_df_original = pd.DataFrame([input_values], columns=features)
        
        # Map Dyslipidemia back to HighCholesterol for model compatibility
        input_df_for_model = input_df_original.copy()
        if 'Dyslipidemia' in input_df_for_model.columns:
            input_df_for_model = input_df_for_model.rename(columns={'Dyslipidemia': 'HighCholesterol'})
        
        # Separate categorical and continuous features
        model_features = list(input_df_for_model.columns)
        continuous_features_in_model = [f for f in model_features if f not in categorical_features]
        
        # Create full feature DataFrame for scaler
        input_full = pd.DataFrame(0.0, index=[0], columns=scaler_features)
        
        # Fill in continuous features
        for col in continuous_features_in_model:
            if col in scaler_features:
                input_full[col] = input_df_for_model[col].values[0]
        
        # Standardize continuous features
        if scaler:
            continuous_standardized = scaler.transform(input_full)
        else:
             return jsonify({'success': False, 'error': 'Scaler not loaded'})

        # Get standardized values
        continuous_standardized_dict = {}
        for col in continuous_features_in_model:
            if col in scaler_features:
                idx = scaler_features.index(col)
                continuous_standardized_dict[col] = continuous_standardized[0, idx]
        
        # Combine standardized continuous features with categorical features
        final_input = []
        for col in model_features:
            if col in categorical_features:
                final_input.append(input_df_for_model[col].values[0])
            else:
                final_input.append(continuous_standardized_dict[col])
        
        input_standardized = np.array([final_input])
        
        # Clip inputs to training data range to avoid extrapolation
        # Note: Input clipping removed since we no longer have full training data
        # The SHAP background data provides sufficient range coverage
        # Make prediction
        prediction = float(predict_fn(input_standardized)[0])
        
        # Determine Risk Level
        model_key = 'all_cause' if model_type == 'all_cause' else 'cardio'
        low_thresh = thresholds[model_key]['low']
        high_thresh = thresholds[model_key]['high']
        
        if prediction < low_thresh:
            risk_level = "Low Risk"
            risk_color = "green"
            risk_desc = "Below half of optimal cutoff"
        elif prediction < high_thresh:
            risk_level = "Medium Risk"
            risk_color = "orange"
            risk_desc = "Intermediate risk zone"
        else:
            risk_level = "High Risk"
            risk_color = "red"
            risk_desc = "Above optimal Youden cutoff"

        # Generate prediction label
        if model_type == 'all_cause':
            prediction_label = f"20-Year All-Cause Mortality Risk: {prediction:.2%}"
            prediction_note = (f"Risk Level: {risk_level}. "
                               "Note: 'High Risk' is defined by the optimal statistical threshold (Youden Index) "
                               "for 20-year mortality.")
        else:
            prediction_label = f"20-Year Cardiovascular Mortality Risk: {prediction:.2%}"
            prediction_note = (f"Risk Level: {risk_level}. "
                               "Note: This estimates the probability of cardiovascular death within 20 years. "
                               "'High Risk' warrants closer cardiovascular surveillance.")
        
        # Calculate SHAP values with higher accuracy
        # nsamples=1000 ensures precision as per methodology
        shap_values = explainer.shap_values(input_standardized, nsamples=1000)
        
        # Handle list output from explainer
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            if len(base_value) > 1:
                base_value = base_value[1] if len(base_value) == 2 else base_value[0]
            else:
                base_value = base_value[0]
        base_value = float(base_value)
        
        # Create display names
        display_names = [feature_mapping.get(f, f) for f in features]
        
        # Create SHAP explanation object
        shap_explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=input_df_original.values[0],
            feature_names=display_names
        )
        
        # Generate waterfall plot
        shap.plots.waterfall(shap_explanation, show=False, max_display=15)
        
        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=1200, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Prepare SHAP contributions for display
        shap_contributions = []
        for i, feat in enumerate(features):
            shap_contributions.append({
                'feature': feature_mapping.get(feat, feat),
                'value': float(input_df_original.iloc[0, i]),
                'shap_value': float(shap_values[0][i])
            })
        
        # Sort by absolute SHAP value
        shap_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': prediction_label,
            'prediction_note': prediction_note,
            'base_value': base_value,
            'shap_plot': image_base64,
            'shap_contributions': shap_contributions
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Starting Web Calculator Application")
    print("="*70)
    print("\nFeature Encoding:")
    print("  - Dyslipidemia: 1 = Yes (have dyslipidemia), 2 = No (no dyslipidemia)")
    print("  - Sex (Gender): 1 = Male, 2 = Female")
    print("  - Smoking: 0 = Never, 1 = Former, 2 = Current")
    print("\nServer starting on http://0.0.0.0:5001")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
