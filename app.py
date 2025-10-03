from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import shap
import os
import json
from datetime import datetime
from datetime import datetime, timedelta
from flask import send_file
from sqlalchemy import func

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)
    input_data = db.Column(db.String(5000), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


try:
    kepler_model = joblib.load('kepler_rf_model.joblib')
    tess_model = joblib.load('tess_xgb_model.joblib')
    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure both model files are in the same directory.")
    exit()


kepler_label_mapping = {
    'CANDIDATE': 'CP', 'FALSE POSITIVE': 'FP', 'CONFIRMED': 'C'
}
tess_label_mapping = {
    0: 'APC', 1: 'CP', 2: 'FA', 3: 'FP', 4: 'KP', 5: 'PC'
}


def save_to_history(model_name, input_data, prediction):
    history_entry = PredictionHistory(
        model_name=model_name,
        input_data=json.dumps(input_data),
        prediction=prediction,
    )
    db.session.add(history_entry)
    db.session.commit()




from sqlalchemy import func 



def calculate_shap_values(model, input_data, required_features):
    """
    Switches to the model-agnostic KernelExplainer and uses a zero-vector baseline
    to generate meaningful, non-zero SHAP values.
    """
    
    import numpy as np
    import pandas as pd
    import shap
    

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=required_features, fill_value=0)


    input_data_safe = np.round(input_df.values, decimals=6)
    

    

    background_data = np.zeros_like(input_data_safe)
    

    if hasattr(model, 'predict_proba'):

        predict_fn = lambda X: model.predict_proba(X)[:, 1]
    else:
      
        predict_fn = lambda X: model.predict(X)
        
  
    explainer = shap.KernelExplainer(predict_fn, background_data)
        
   
    shap_values = explainer.shap_values(input_data_safe, silent=True)
    
    
    abs_shap_array = np.abs(np.squeeze(shap_values))

 
    abs_shap_values = abs_shap_array.ravel()
    
  
    features = input_df.columns.tolist()
    explanation_data = []
    
  
    if len(features) != len(abs_shap_values):
        raise ValueError(f"Feature count mismatch after SHAP calculation: {len(features)} != {len(abs_shap_values)}")

    for feature, value in zip(features, abs_shap_values):
        explanation_data.append({
            'feature': feature,
            'importance': float(value) 
        })
            
   
    explanation_data.sort(key=lambda x: x['importance'], reverse=True)
    
    return explanation_data




@app.route('/explain_prediction', methods=['POST'])
def explain_prediction_route():
    try:
        data = request.json
        model_name = data.get('model')
        input_data = data.get('input_data') 
        
        model_name_upper = model_name.upper() if model_name else ''
        
        
        if model_name_upper == 'KEPLER': 
            model = kepler_model
            required_features = ['kepid', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']
        
        elif model_name_upper == 'TESS': 
            model = tess_model
            required_features = ['toi', 'tid', 'ra', 'dec', 'st_pmra', 'st_pmraerr1', 'st_pmraerr2', 'st_pmralim', 'st_pmdec', 'st_pmdecerr1', 'st_pmdecerr2', 'st_pmdeclim', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_tranmidlim', 'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandurhlim', 'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2', 'pl_trandeplim', 'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 'pl_radelim', 'pl_insol', 'pl_eqt', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2', 'st_tmaglim', 'st_dist', 'st_disterr1', 'st_disterr2', 'st_distlim', 'st_teff', 'st_tefferr1', 'st_tefferr2', 'st_tefflim', 'st_logg', 'st_loggerr1', 'st_loggerr2', 'st_logglim', 'st_rad', 'st_raderr1', 'st_raderr2', 'st_radlim']
        
        else:
            return jsonify({'error': 'Invalid model specified for explanation.'}), 400

        
        explanation_data = calculate_shap_values(model, input_data, required_features)
        
        return jsonify({'explanation': explanation_data})

    except Exception as e:
        
        print(f"SHAP explanation failed: {e}")
        return jsonify({'error': f'Failed to generate explanation. Detail: {str(e)}'}), 500


@app.route('/')
def index():
    return render_template('index.html')

def cleanup_duplicate_predictions(model, input_data):
    """
    Finds and deletes all but the oldest PredictionHistory entry 
    that shares the exact same model and input data within a short time window.
    """
    
    
    time_window = datetime.now() - timedelta(seconds=5)

    
    potential_duplicates = PredictionHistory.query.filter(
        PredictionHistory.model == model,
        PredictionHistory.input_data == input_data,
        PredictionHistory.timestamp >= time_window
    ).order_by(PredictionHistory.timestamp.asc()).all()


    if len(potential_duplicates) > 1:
       
        duplicates_to_delete = potential_duplicates[1:] 
        
        
        for entry in duplicates_to_delete:
            db.session.delete(entry)
            print(f"CLEANUP WORKER: Deleted duplicate ID {entry.id} for model {model}.")
        
        try:
            db.session.commit()
            print(f"CLEANUP WORKER: Successfully committed deletion of {len(duplicates_to_delete)} duplicate(s).")
        except Exception as e:
            db.session.rollback()
            print(f"Database error during cleanup: {e}")

@app.route('/download_sample/<model_type>')
def download_sample(model_type):
    """Serves the appropriate sample CSV file for download."""
   
    if model_type == 'kepler':
        file_path = 'kepler_sample.csv'
        filename = 'kepler_bulk_sample.csv'
    elif model_type == 'tess':
        file_path = 'tess_sample.csv'
        filename = 'tess_bulk_sample.csv'
    else:
        return jsonify({"error": "Invalid model type for sample."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": f"Sample file not found: {file_path}"}), 500

    return send_file(
        file_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )



@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    """Handles the file upload, prediction, and saving for bulk input."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part found in request.'}), 400

    file = request.files['file']
    model_name = request.form.get('model')

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if model_name not in ['kepler', 'tess']:
        return jsonify({'error': 'Invalid model specified for bulk prediction.'}), 400

    try:
        
        df = pd.read_csv(file)
        
        
        if model_name == 'kepler':
            model = kepler_model
            label_mapping = kepler_label_mapping
           
            required_features = ['kepid', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period', 'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2', 'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth', 'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1', 'koi_prad_err2', 'koi_teq', 'koi_insol', 'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2', 'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad', 'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']
        else:
            model = tess_model
            label_mapping = tess_label_mapping
           
            required_features = ['toi', 'tid', 'ra', 'dec', 'st_pmra', 'st_pmraerr1', 'st_pmraerr2', 'st_pmralim', 'st_pmdec', 'st_pmdecerr1', 'st_pmdecerr2', 'st_pmdeclim', 'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_tranmidlim', 'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandurhlim', 'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2', 'pl_trandeplim', 'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 'pl_radelim', 'pl_insol', 'pl_eqt', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2', 'st_tmaglim', 'st_dist', 'st_disterr1', 'st_disterr2', 'st_distlim', 'st_teff', 'st_tefferr1', 'st_tefferr2', 'st_tefflim', 'st_logg', 'st_loggerr1', 'st_loggerr2', 'st_logglim', 'st_rad', 'st_raderr1', 'st_raderr2', 'st_radlim']

        
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
             return jsonify({'error': f'CSV structure error: Missing expected columns: {", ".join(missing_cols)}. Please use the provided sample format.'}), 400
        
        
        df_clean = df[required_features].dropna(how='any')

        
        if df_clean.empty:
            return jsonify({'error': 'The CSV is empty, or all data rows contained missing values after strict cleaning.'}), 400

        
        print(f"--- Bulk Predict Debug ---")
        print(f"Original DataFrame size: {len(df.index)} rows")
        print(f"Cleaned DataFrame size for prediction: {len(df_clean.index)} rows")
        print(f"--------------------------")
        
        
        predictions_encoded = model.predict(df_clean)
        
        
        for index, row in df_clean.iterrows():
            
            encoded = predictions_encoded[index]
            predicted_label = label_mapping.get(encoded, "Unknown")
            input_data_dict = row.to_dict()
            save_to_history(model_name.capitalize(), input_data_dict, predicted_label)
            delete_all_old_duplicate_predictions()
            
            
        delete_all_old_duplicate_predictions()
        return jsonify({'message': f'Bulk prediction completed and saved. ({len(df_clean.index)} rows predicted)'}), 200

    except KeyError as e:
        return jsonify({'error': f'CSV structure error: Missing or misnamed column {e}. Please use the provided sample format.'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing failed. Please check the file format and data types. Detail: {str(e)}'}), 500

@app.route('/reset_history', methods=['POST'])
def reset_history():
    """Deletes all entries from the prediction history table."""
    try:
        
        db.session.query(PredictionHistory).delete()
        db.session.commit()
        return jsonify({"message": "History reset successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/predict_kepler', methods=['POST'])
def predict_kepler():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        
       
        prediction = kepler_model.predict(input_df)[0]
        prediction_label = kepler_label_mapping.get(prediction, "Unknown")
        
       
        save_to_history('Kepler', data, prediction_label)
        
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_tess', methods=['POST'])
def predict_tess():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        
        
        prediction_encoded = tess_model.predict(input_df)[0]
        prediction_label = tess_label_mapping.get(prediction_encoded, "Unknown")
        
      
        save_to_history('TESS', data, prediction_label)
        
        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def delete_all_old_duplicate_predictions():
    """
    Finds the newest ID for each unique prediction and deletes all older duplicates.
    Fixes: Corrects PredictionHistory.model to PredictionHistory.model_name.
    """
    try:
        
        entries_to_keep_ids = db.session.query(
            func.max(PredictionHistory.id)
        ).group_by(
            PredictionHistory.model_name, 
            PredictionHistory.input_data,
            PredictionHistory.prediction
        ).all()
        
        
        keep_ids = [id_[0] for id_ in entries_to_keep_ids]

        
        deleted_count = db.session.query(PredictionHistory).filter(
            PredictionHistory.id.notin_(keep_ids)
        ).delete(synchronize_session=False)

        db.session.commit()
        print(f"BRUTE-FORCE CLEANER: Successfully deleted {deleted_count} duplicate predictions.")
        return deleted_count
    except Exception as e:
        db.session.rollback()
        print(f"Database error during brute-force cleanup: {e}")
        
        return 0

@app.route('/history', methods=['GET'])
def get_history():
    delete_all_old_duplicate_predictions()
    search_query = request.args.get('search', '')
    filter_model = request.args.get('filter', 'all')
    
    query = PredictionHistory.query
    
    if filter_model != 'all':
        query = query.filter_by(model_name=filter_model)
    
    if search_query:
        query = query.filter(PredictionHistory.input_data.like(f'%{search_query}%') |
                             PredictionHistory.prediction.like(f'%{search_query}%'))
    
    history_data = query.order_by(PredictionHistory.timestamp.desc()).all()
    
    history_list = []
    for entry in history_data:
        history_list.append({
            'id': entry.id,
            'model_name': entry.model_name,
            'input_data': json.loads(entry.input_data),
            'prediction': entry.prediction,
            'timestamp': entry.timestamp.strftime('%Y-%m-%d')
        })
    
    return jsonify(history_list)

if __name__ == '__main__':
    
    with app.app_context():
        db.create_all()
    app.run(debug=True)