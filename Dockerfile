# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to match the docker-compose volume path
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run pyssem.py when the container launches
CMD ["python", "./pyssem/pyssem.py"]