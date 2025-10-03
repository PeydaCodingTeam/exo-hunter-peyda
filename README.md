# 🌌 NASA Exoplanet Predictor
A machine learning web application for classifying potential exoplanets using data from NASA's Kepler and TESS missions. This tool helps astronomers and researchers analyze celestial objects and predict whether they are confirmed exoplanets, candidates, or false positives.

# 🚀 Features

🔭 Prediction Capabilities
Dual Model System: Separate trained models for Kepler and TESS mission data

Single Prediction: Real-time classification for individual celestial objects

Bulk Processing: CSV batch predictions for large datasets

SHAP Explanations: Model interpretability with feature importance analysis

# 📊 Model Information
Kepler Model: Random Forest classifier (92.5% accuracy)

TESS Model: XGBoost multi-class classifier (72% accuracy)

Comprehensive Classification: Distinguishes between confirmed planets, candidates, and various false positive types

# 💾 Data Management
Prediction History: Complete audit trail of all predictions

Search & Filter: Find predictions by model, date, or results

Export Capabilities: Download history as CSV for further analysis

Duplicate Prevention: Smart cleanup of redundant entries

# 🛠 How To Run
Install Python 3.11.0 or higher

Go to directory of the project

open terminal

pip install -r requirements.txt

py app.py

Open browser and go to http://127.0.0.1:5000/





# 🎯 Usage Guide
Select Model: Choose between Kepler or TESS

Input Data: Fill in the required feature values

Get Results: Click "See Results" for prediction

Explain: Use "Explain" button for SHAP feature importance

Bulk Prediction: Download Sample

Prepare Data: Fill template with your observation data

Upload: Use "Bulk Input" to process multiple predictions

Review: Check History page for results

# 📜 Prediction History
Tabs: All Models, Kepler, TESS

Export to CSV: Download filtered results as CSV

Reset history: Clear entire history (irreversible)

View Details: See inputs of the prediction

Explain: See the influence of each input

# 🔬 Model Details
Kepler Mission Model Algorithm: Random Forest Classifier

Kepler Model Accuracy: 92.5%

Training Data: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

Classifications:

CONFIRMED (C): Verified exoplanets

CANDIDATE (CP): Potential planets needing verification

FALSE POSITIVE (FP): Non-planetary signals

TESS Mission Model Algorithm: XGBoost Multi-Class Classifier

TESS Model Accuracy: 72%

Training Data: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI

Classifications:

PC: Planet Candidate

APC: Astrophysical Planet Candidate

EB: Eclipsing Binary

V: Variable Star

FP: False Positive

# 📊 Feature Variables
Kepler Features (40+ parameters)
kepid, koi_score, koi_period, koi_depth

koi_teq, koi_insol, koi_steff, koi_slogg

ra, dec, koi_kepmag, and more...

TESS Features (50+ parameters)
toi, tid, ra, dec, pl_orbper

pl_trandep, pl_rade, pl_insol, st_teff

st_logg, st_rad, st_tmag, and more...

# 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_kepler` | POST | Single Kepler prediction |
| `/predict_tess` | POST | Single TESS prediction |
| `/bulk_predict` | POST | Batch CSV predictions |
| `/explain_prediction` | POST | SHAP explanation |
| `/history` | GET | Prediction history |
| `/reset_history` | POST | Clear all history |
| `/download_sample/<model>` | GET | Download CSV templates |


# 🎨 Interface Overview
Main Sections

Predict: Input forms for single and bulk predictions

History: Complete prediction log with search/filter

Variables: Detailed documentation of all features

Model Info: Technical specifications and accuracy metrics

UI Features
Dark Theme: Space-inspired color scheme

Responsive Design: Works on desktop and mobile

Real-time Feedback: Immediate prediction results

Modal Dialogs: Clean data presentation

Interactive Tables: Sortable and searchable history

# 🚨 Troubleshooting

Port Already in Use:

Kill existing process or use different port

lsof -ti:5000 | xargs kill -9

CSV Upload Errors:

Verify column names match sample files

Ensure no missing required values

Check for proper numeric formatting

# 🙏 Acknowledgments
NASA Exoplanet Archive for training data

Kepler and TESS mission teams

Scikit-learn and XGBoost communities

SHAP for model interpretability

# ❓ About Us

Our team, Peyda, founded in 2025 by Foad Fereidooni and Rambod Roshani, is a creative software team. Both founders are members of the International Federation of Inventors’ Associations (IFIA). The team specializes in responsive websites, modern UI/UX, and AI-driven solutions, offering web & mobile development, digital marketing, and content creation to deliver secure, scalable, and innovative technology.

You can get in contact with us via these Emails.

iliya.fereydouni@gmail.com

ram89.bod@gmail.com
