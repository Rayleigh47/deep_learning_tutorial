build_image() {
    echo "Building Docker image..."
    docker build -t pytorch-gpu . -f Dockerfile
}

# Function to run the Docker container
run_container() {
    echo "Running Docker container..."
    docker run --name pytorch-container --gpus all -it --rm \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    pytorch-gpu
}

# Help message
usage() {
    echo "Usage: $0 {build|run}"
    exit 1
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    usage
fi

# Parse the command-line argument
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    *)
        usage
        ;;
esac
