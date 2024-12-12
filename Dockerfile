# Use the official Python runtime as a parent image

FROM python:3.9-slim

# Set the working directory to /app

# Set the working directory in the container
WORKDIR /Web-Browser

COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt

RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container

COPY . .

EXPOSE 5000

# Run app.py when the container launches

CMD ["python", "app.py"]