from flask import Flask, render_template, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import shap
import os
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from pytrends.request import TrendReq
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import time


load_dotenv()

app = Flask(__name__)

try:
    model = load_model('model/best_model.h5')
    le = joblib.load('Mapping/label_encoder_kategori.pkl')
    scaler = joblib.load('Mapping/scaler_startup_success.pkl')
    
    background = np.zeros((10, len(scaler.feature_names_in_) + 1))
    def model_predict(X):
        return model.predict(X).flatten()
    explainer = shap.KernelExplainer(model_predict, background)
    PREDICTOR_LOADED = True
except (IOError, FileNotFoundError):
    PREDICTOR_LOADED = False

try:
    vertexai.init(project=os.getenv("VERTEX_PROJECT_ID"), location="us-central1")
    chat_model = GenerativeModel(os.getenv("VERTEX_ENDPOINT_ID"))
    CHAT_MODEL_LOADED = True
except Exception:
    CHAT_MODEL_LOADED = False

safety_settings = [
    SafetySetting(category=cat, threshold=SafetySetting.HarmBlockThreshold.OFF)
    for cat in SafetySetting.HarmCategory
]

def calculate_age_in_years(date_str):
    today = datetime.today()
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        delta = today - date_obj
        return round(delta.days / 365, 2)
    except (ValueError, TypeError):
        return 0.0

def generate_reason_with_values(shap_vals, features, label_output, original_values):
    display_labels = {
        'umur_milestone_terakhir': 'Last Milestone Age', 'relasi': 'Number of Relations/Investors',
        'umur_pendanaan_pertama': 'First Funding Age', 'total_dana': 'Total Funds',
        'umur_pendanaan_terakhir': 'Last Funding Age', 'umur_milestone_pertama': 'First Milestone Age',
        'rata_partisipan': 'Average Participants/Customers', 'jumlah_pendanaan': 'Number of Funding Rounds',
        'jumlah_milestone': 'Number of Milestones', 'rasio_dana_per_relasi': 'Fund to Relationship Ratio',
        'dana_per_pendanaan': 'Average Funding per Round', 'populer': 'Popularity Status'
    }
    positive_reasons, negative_reasons = [], []
    shap_list = sorted(list(zip(features, shap_vals[0])), key=lambda x: abs(x[1]), reverse=True)

    for feat, val in shap_list:
        nilai_asli = original_values.get(feat, 0)
        unit = " years" if "umur" in feat else ""
        if isinstance(nilai_asli, str):
            nilai_fmt = nilai_asli
        elif abs(nilai_asli) >= 1_000_000:
            nilai_fmt = f"{int(nilai_asli / 1_000_000):,} million"
        else:
            nilai_fmt = f"{int(nilai_asli):,}"
        
        label_feat = display_labels.get(feat, feat.replace('_', ' '))
        if label_output.lower() == 'failure':
            direction_text = 'reduces the risk of failure' if val > 0 else 'increases the risk of failure'
        else:
            direction_text = 'supports the potential for success' if val > 0 else 'reduces the potential for success'
        
        reason_text = f"'{label_feat}' ({nilai_fmt}{unit}) {direction_text}"
        if val > 0:
            positive_reasons.append(reason_text)
        else:
            negative_reasons.append(reason_text)
    return positive_reasons, negative_reasons

def predict_and_explain(data_dict):
    df = pd.DataFrame([data_dict])
    df['kategori_encoded'] = le.transform(df['kategori'])
    original_input = df.copy()
    df['kategori'] = df['kategori_encoded']
    df.drop(columns=['kategori_encoded'], inplace=True)
    scaled_features = scaler.transform(df[scaler.feature_names_in_])
    sample_scaled = np.concatenate([scaled_features, df[['populer']].values], axis=1)
    prediction = model.predict(sample_scaled)
    binary_result = int(prediction[0][0] > 0.5)
    label = "Success" if binary_result == 1 else "Failure"
    shap_values = explainer.shap_values(sample_scaled)
    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
    scaled_part = sample_scaled[0][:-1].reshape(1, -1)
    inversed_scaled = scaler.inverse_transform(scaled_part).flatten()
    inverse_values = dict(zip(scaler.feature_names_in_, inversed_scaled))
    inverse_values['populer'] = int(df['populer'].values[0])
    inverse_values['kategori'] = original_input['kategori'].values[0]
    pos_reason, neg_reason = generate_reason_with_values(
        shap_array, list(scaler.feature_names_in_) + ['populer'], label, inverse_values
    )
    return label, pos_reason, neg_reason

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    if request.method == 'GET':
        return render_template('predictor.html')
    if request.method == 'POST':
        if not PREDICTOR_LOADED:
            return render_template('success.html',
                                   result="Demo Mode",
                                   pos_reason=["High number of investors increases success chance."],
                                   neg_reason=["Low funding per round decreases success chance."])
        try:
            data_input = {
                'umur_milestone_terakhir': calculate_age_in_years(request.form.get('tanggal_pencapaian_terakhir')),
                'relasi': int(request.form.get('relasi')),
                'umur_pendanaan_pertama': calculate_age_in_years(request.form.get('tanggal_pendanaan_pertama')),
                'total_dana': float(request.form.get('total_dana')),
                'umur_pendanaan_terakhir': calculate_age_in_years(request.form.get('tanggal_pendanaan_terakhir')),
                'umur_milestone_pertama': calculate_age_in_years(request.form.get('tanggal_pencapaian_awal')),
                'rata_partisipan': int(request.form.get('rata_partisipan')),
                'kategori': request.form.get('kategori'),
                'jumlah_pendanaan': int(request.form.get('jumlah_pendanaan')),
                'jumlah_milestone': int(request.form.get('jumlah_capaian')),
                'rasio_dana_per_relasi': float(request.form.get('rasio_dana_per_relasi')),
                'dana_per_pendanaan': float(request.form.get('dana_per_pendanaan')),
                'populer': int(request.form.get('populer'))
            }
            result, pos_reason, neg_reason = predict_and_explain(data_input)
            return render_template('success.html', result=result, pos_reason=pos_reason, neg_reason=neg_reason)
        except Exception as e:
            return f"An error occurred: {e}", 500

@app.route('/forecaster', methods=['GET', 'POST'])
def forecaster():
    if request.method == 'POST':
        keywords_string = request.form.get('keywords')
        if not keywords_string or not keywords_string.strip():
            return render_template('forecaster.html', error="Input cannot be empty. Please enter at least one keyword.")

        temp_list = [keyword.strip() for keyword in keywords_string.split(',')]
        keywords_list = [keyword for keyword in temp_list if keyword]

        if not keywords_list:
            return render_template('forecaster.html', error="Invalid input. Please enter valid keywords.")
        
        keywords_list = keywords_list[:5]

        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(keywords_list, cat=0, timeframe='today 5-y', geo='', gprop='')
            interest_df = pytrends.interest_over_time()

            if 'isPartial' in interest_df.columns:
                interest_df = interest_df.drop(columns=['isPartial'])

            if interest_df.empty:
                return render_template('forecaster.html', error="Could not retrieve data. The keywords might be too specific or have no search volume.")
            
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(12, 6))
            interest_df.plot(kind='line', ax=ax, lw=2, alpha=0.9)
            
            ax.set_title('Strategic Trend Comparison: Market Interest', fontsize=16)
            ax.set_ylabel('Relative Search Interest (0-100)')
            ax.set_xlabel('Date')
            ax.grid(True, which='major', linestyle='--', linewidth=0.6)
            ax.legend(title='Investment Areas')
            
            plot_path = os.path.join('static', 'images', 'trends_chart.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            
            return render_template('forecaster.html', plot_image=plot_path, timestamp=int(time.time()))

        except Exception as e:
            error_msg = f"An error occurred. This could be due to a temporary issue with Google Trends or your network. (Error: {e})"
            return render_template('forecaster.html', error=error_msg)
            
    return render_template('forecaster.html')

@app.route('/consultant')
def consultant():
    return render_template('consultant.html')

@app.route('/generate_chat', methods=['POST'])
def generate_chat():
    if not CHAT_MODEL_LOADED:
        return jsonify({'response': 'Sorry, the chatbot model is currently unavailable.'})

    try:
        data = request.get_json()
        user_input = data['user_input']
        
        system_instruction = (
            "You are Viola, a risk management consultant tasked with helping banks and investors make "
            "informed decisions about loans and funding, with a focus on evaluating the risk of startups, individuals, or institutions. "
            "You must only answer questions related to risk management, credit analysis, due diligence, and investment feasibility assessment. "
            "If a question is outside these topics, respond with: "
            "'I'm sorry, I can only assist with questions about risk management, credit, and investment evaluation.'\n"
        )
        prompt = system_instruction + user_input
        
        chat_session = chat_model.start_chat()
        response = chat_session.send_message(prompt, safety_settings=safety_settings)
        
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)