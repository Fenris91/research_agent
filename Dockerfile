FROM python:3.9-slim
WORKDIR /app
# Copy data (CSVs) into container
COPY data/ data/
# Copy notebook scripts into container
COPY notebooks/ notebooks/
# Install requirements
RUN pip install pandas plotly
# Set entrypoint to run script
CMD ["python", "notebooks/time_series_anthropology.py"]
