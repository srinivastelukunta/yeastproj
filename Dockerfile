# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE $PORT

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app 