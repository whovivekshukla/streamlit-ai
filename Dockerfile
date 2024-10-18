# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a shell script to load environment variables
RUN echo '#!/bin/sh\n\
export $(grep -v "^#" .env | xargs -d "\n")\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose the port that Streamlit will run on
EXPOSE 8501

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]