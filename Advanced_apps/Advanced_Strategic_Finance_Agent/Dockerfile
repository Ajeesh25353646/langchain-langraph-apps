# Use an official lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user

# Create a non-root user (Hugging Face recommendation)
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Set path for user-installed binaries
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy requirements and install dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Expose the default port for Hugging Face Spaces (Docker SDK)
EXPOSE 7860

# Command to run the application
# We bind to 0.0.0.0 and port 7860 as required by HF Spaces
CMD ["streamlit", "run", "Strategic_Finance_Agent.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
