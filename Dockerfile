# Use official Python image
FROM python:3.10-slim  

# Set the working directory
WORKDIR /app  

# Copy requirements and install dependencies
COPY Requirements.txt .  
RUN pip install --no-cache-dir -r Requirements.txt --timeout=300  

COPY *.csv /app/

# Copy application files
COPY . .  

# Expose the FastAPI default port
EXPOSE 8000  

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
