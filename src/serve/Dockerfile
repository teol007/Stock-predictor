FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Run the FastAPI app (specify the subfolder if main.py is in src/)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
