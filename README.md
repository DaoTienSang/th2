===== README.md =====

Chay local:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data_pipeline.py
python train.py
python registry.py
uvicorn app:app --reload --port 8000
```

Test API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body (Get-Content models\sample_input.json -Raw)
```

MLflow UI:

```powershell
mlflow ui --backend-store-uri mlruns --port 5000
```

Test:

```powershell
pytest -q
```
