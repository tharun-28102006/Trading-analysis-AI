FROM python:3.12-slim
WORKDIR /srv/nivesh
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY README.md ./README.md
ENV NIVESH_HOST=0.0.0.0 NIVESH_PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host ${NIVESH_HOST} --port ${NIVESH_PORT}"]
