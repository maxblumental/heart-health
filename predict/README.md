Basic Docker commands:

1) build an image with a tag:

```
docker build -t predict:1.0 .
```

2) run the image inside a container (`-p` or `--publish` maps the host's port to the container's port):

```
docker run -p 8000:8000 predict:1.0
```