PYTHON = C:\Users\Admin\Downloads\APPLICATION\python_env\python.exe

.PHONY: install data train evaluate app test lint typecheck audit clean

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train_models.py

evaluate:
	$(PYTHON) scripts/evaluate.py

app:
	$(PYTHON) -m streamlit run app.py --server.address=0.0.0.0 --server.headless=true

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src

audit:
	$(PYTHON) scripts/audit.py

clean:
	powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue __pycache__, .pytest_cache, .mypy_cache, .ruff_cache, *.egg-info"
