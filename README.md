# Air Quality Index API

Predict the polluted gas (CO, SO2, NO, ...) and PM2.5 in Ho Chi Minh City using Deep Learning models

## How to run the server locally ?

This requires **poetry** pip packages to be installed, then all poetry dependencies

```
poetry install
make run-local
```

## How to build and run server ?

```
make build-server
make run-server
```

The above commands will build and run server with default values for below options

Available options:

* PORT (default: 8000)
* IMAGE_NAME (default: air_quality_index_api)
* IMAGE_TAG (default: latest)
* CONTAINER_NAME (default: air_quality_index_api)

Example:

```
make build-server PORT=8080 IMAGE_NAME=test IMAGE_TAG=test
make run-server CONTAINER_NAME=testcontainer PORT=8080 IMAGE_NAME=test IMAGE_TAG=test
```

## Online repository

The Docker image is available at: <https://hub.docker.com/repository/docker/tphiepbk/air_quality_index_api/general>

## API testing

Please refer to [APIs_testing](./APIs_testing.md) to see all the CURL commands with sample data
