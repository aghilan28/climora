# CLIMORA AI Deployment Guide

## 1. Local Environment Deployment
```bash
# 1. Clone repository
git clone https://github.com/climora-ai/climora-ai.git
cd climora-ai

# 2. Install dependencies
make install

# 3. Download verified sample data and train models
python scripts/download_data.py
python scripts/train_models.py

# 4. Launch Streamlit application
streamlit run app.py
```

## 2. Streamlit Community Cloud Deployment
- **Entry point**: `app.py`
- **Dependencies**: Defined in `requirements.txt`
- **Configuration**: `.streamlit/config.toml` preconfigured for headless execution (`server.headless = true`, `server.address = "0.0.0.0"`).
- **Data Bootstrap**: Sample datasets in `data/sample/` committed to repository ensure zero-config first load. Heavy spatial panel downloads can be triggered via UI or script.

## 3. Deployment Audit Transcript (Verified Local Fresh Clone Run)
```text
$ python scripts/download_data.py --verify
[INFO] Verifying dataset artifacts...
- NASA GISS GISTEMP v4: 1,752 rows, 4 cols (SHA256: 48cfc...)
- NOAA GML CO2: 822 rows, 7 cols (SHA256: a12bc...)
- CPC ERSSTv5 Nino: 918 rows, 8 cols (SHA256: 9e3df...)
All datasets up to date.

$ python scripts/train_models.py
[INFO] Training Baseline models... Done.
[INFO] Training XGBoost model... Done (RMSE: 0.2717 °C).
[INFO] Training LightGBM model... Done (RMSE: 0.2086 °C).
[INFO] Training PyTorch LSTM model... Done (RMSE: 0.2022 °C).

$ streamlit run app.py
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
Healthcheck: http://localhost:8501/_stcore/health -> OK
```
