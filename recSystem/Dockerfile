FROM python:3.9-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Expose the port that the application listens on.
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=recSystem.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run app.py when the container launches
CMD ["python3", "recSystem.py"]
