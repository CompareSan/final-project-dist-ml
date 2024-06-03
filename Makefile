pre-commit:
	poetry run pre-commit run --all-files

test:
	poetry run pytest -x -vv -n 2

mlflow:
	cd final_project_dist_ml && mlflow server --backend-store-uri sqlite:///backend.db

backend:
	cd backend && poetry run uvicorn app:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && streamlit run streamlit_app.py


.PHONY: pre-commit test mlflow backend frontend
