# Use a lightweight Python base image
FROM python:3.11-slim
 
# Set working directory inside the container
WORKDIR /app
 
# Copy dependency list first (this layer gets cached, speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the rest of the app (app.py + your .pkl files)
COPY . .
 
# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000
 
# Start the server when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]