FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -e .[dev]

# Copy project
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash samsaek
RUN chown -R samsaek:samsaek /app
USER samsaek

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "samsaek.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]