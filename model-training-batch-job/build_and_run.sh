# Build and run script for model training batch job

# Build the Docker image
echo "Building Docker image..."
docker build -t model-training-job .

# Check if the container is already running
if [ "$(docker ps -q -f name=model-training-job)" ]; then
    echo "Stopping existing container..."
    docker stop model-training-job
    docker rm model-training-job
fi

# Check if the ml-network exists, create if not
if ! docker network inspect ml-network &>/dev/null; then
  echo "Creating ml-network Docker network..."
  docker network create ml-network
fi

# Run the container
echo "Starting container..."
docker run -d --name model-training-job \
    --network ml-network \
    -e DB_USER=${DB_USER:-postgres} \
    -e DB_PASSWORD=${DB_PASSWORD} \
    -e DB_HOST=${DB_HOST:-db} \
    -e TRAINING_INTERVAL=${TRAINING_INTERVAL:-24} \
    model-training-job

echo "Container started!"
echo "To view logs, run: docker logs -f model-training-job" 