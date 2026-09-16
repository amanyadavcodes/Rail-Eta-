# Minimal Rail ETA

## Install

```bash
python -m pip install -r requirements.txt
```

### Windows with Python 3.13

Run the included setup script from PowerShell:

```powershell
.\setup_windows_313.ps1
```

Then run:

```powershell
.\venv\Scripts\python.exe live_predict.py 12919
```

The application code supports Python 3.13. The separate
`requirements-py313.txt` ensures that pip installs package releases which
provide Python 3.13 binary wheels.

Copy `.env.example` to `.env` and set your private RailRadar key:

```env
API_KEY=your_actual_key
```

## Train

Place `final_eta_dataset.csv` at
`data/processed/final_eta_dataset.csv`, then run:

```bash
python train.py
```

## Make a live prediction

```bash
python live_predict.py 12919
```

Optional journey start date:

```bash
python live_predict.py 12919 --date 2026-09-16
```

The program calls RailRadar's documented
`GET /v1/trains/{number}/live` endpoint, obtains matching hourly weather from
Open-Meteo, builds the saved XGBoost feature schema, and prints the remaining
ETA and predicted destination-arrival timestamp.