FROM python:3.12-slim

# Get the uv binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install dependencies first, so Docker can catch this layer
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# Use the project's virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy only what the service needs at runtime
COPY predict.py model.joblib ./

EXPOSE 9696

#CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]