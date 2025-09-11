run:
	poetry run uvicorn air_quality_index_api.server:app --app-dir src --reload

