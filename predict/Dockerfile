FROM python:3.9
RUN pip install pandas uvicorn fastapi scikit-learn==0.24.1
COPY models /models
COPY server.py /server.py

WORKDIR .

ENV MODEL_PATH="/models"

ENTRYPOINT ["python", "server.py"]
