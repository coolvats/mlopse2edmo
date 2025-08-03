# Use a lightweight Python image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model and the serving script
COPY model/gradient_boosting_regressor_model.pkl ./model/
COPY serve_model.py .

# Copy the data directory (if needed for preprocessing in serve_model.py, though typically not for inference)
# If your serve_model.py relies on Advertising.csv for preprocessing, uncomment the line below
# COPY data/Advertising.csv ./data/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]