# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for pyaudio and build tools
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*


# Set environment variables

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies with increased timeout and using a different mirror
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Install pyaudio
RUN pip install pyaudio

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "test.py", "--server.port=8501", "--server.address=0.0.0.0"]