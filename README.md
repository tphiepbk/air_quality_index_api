# How to run the server locally ?

This requires **poetry** pip packages to be installed, then all poetry dependencies

```
poetry install
make run-local
```

# How to build and run server ?

```
make build-server
make run-server
```

Available options:

* PORT
* IMAGE_NAME
* IMAGE_TAG
* CONTAINER_NAME

Example:

```
make build-server PORT=8080 IMAGE_NAME=test IMAGE_TAG=test
make run-server CONTAINER_NAME=testcontainer PORT=8080 IMAGE_NAME=test IMAGE_TAG=test
```
