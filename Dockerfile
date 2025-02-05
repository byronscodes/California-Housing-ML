# filepath: /c:/Users/darin/Code/California Housing ML/Dockerfile
# Use an official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install virtualenv and create a virtual environment
RUN python -m venv /app/venv

# Activate virtual environment and install dependencies
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install numpy pandas scikit-learn flask

# Copy project files into the container
COPY . .

# Set the default shell to use the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Set the default command to run the app
CMD ["python", "app.py"]