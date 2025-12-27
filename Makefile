PORT = 8000
IMAGE_NAME = "tphiepbk/air_quality_index_api"
IMAGE_TAG = "1.3"
CONTAINER_NAME = "air_quality_index_api"

run-local:
	poetry run uvicorn src.air_quality_index_api.server:app --host 0.0.0.0 --port $(PORT) --reload

build-server:
	docker build --build-arg PORT=$(PORT) -t $(IMAGE_NAME):$(IMAGE_TAG) .

run-server:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME):$(IMAGE_TAG)

