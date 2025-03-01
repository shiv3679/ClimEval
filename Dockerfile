# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit explicitly
RUN pip install streamlit

# Copy the app files
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Run the app and print helpful instructions using JSON array syntax
CMD ["sh", "-c", "echo '✅ Streamlit App is Running! Access it via: http://localhost:8502' && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
