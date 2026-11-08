FROM python:3.13-slim

WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir --no-deps .
COPY app app

# The adapter repo is private: pass HF_TOKEN at run time, never at build time.
ENV HF_HOME=/srv/cache
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
