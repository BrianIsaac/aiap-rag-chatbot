#!/bin/bash

set -e  # Exit on error

echo "===== RAG System Setup ====="

QDRANT_DIR="corpus/qdrant_data"
OLLAMA_STARTED=false

# --- Utility: Pull model if not already present ---
pull_model_if_missing() {
    local model_name=$1
    if sudo docker exec ollama ollama list | grep -q "$model_name"; then
        echo "✅ Model '$model_name' is already pulled."
    else
        echo "📦 Pulling model '$model_name'..."
        sudo docker exec ollama ollama pull "$model_name"
    fi
}

# --- Step 1: Corpus Handling ---
if [[ -d "$QDRANT_DIR" ]]; then
    echo "⚠️  Found existing corpus at '$QDRANT_DIR'."
    read -p "Do you want to create a new corpus (overwrite existing index)? (y/n): " recreate
    if [[ "$recreate" == "y" ]]; then
        echo "You have chosen to recreate the corpus."
        echo "New documents will be embedded and stored in: $QDRANT_DIR"
        read -p "Are you sure you want to continue? (y/n): " confirm_recreate
        if [[ "$confirm_recreate" == "y" ]]; then
            echo ""
            echo "⚠️  RAG service requires minimum 8GB VRAM."
            echo "For hardware/image requirements, visit: https://hub.docker.com/r/ollama/ollama"
            echo "🔄 Starting Ollama service for preprocessing..."
            sudo docker start ollama
            OLLAMA_STARTED=true
            pull_model_if_missing "nomic-embed-text"
            echo "Running corpus preprocessing..."
            python3 src/preprocess_corpus.py
            echo "✅ Corpus preprocessing complete."
        else
            echo "❌ Aborted corpus recreation. Exiting."
            exit 0
        fi
    else
        echo "✅ Keeping existing corpus."
    fi
else
    echo "📁 No existing corpus found at '$QDRANT_DIR'."
    echo "A new corpus will be created at this path."
    read -p "Do you want to continue and create it? (y/n): " confirm_create
    if [[ "$confirm_create" == "y" ]]; then
        echo ""
        echo "⚠️  RAG service requires minimum 8GB VRAM."
        echo "For hardware/image requirements, visit: https://hub.docker.com/r/ollama/ollama"
        echo "🔄 Starting Ollama service for preprocessing..."
        sudo docker start ollama
        OLLAMA_STARTED=true
        pull_model_if_missing "nomic-embed-text"
        echo "Running corpus preprocessing..."
        python3 src/preprocess_corpus.py
        echo "✅ Corpus preprocessing complete."
    else
        echo "❌ Corpus creation cancelled. Exiting setup."
        exit 0
    fi
fi

# Stop Ollama if it was started for corpus
if [[ "$OLLAMA_STARTED" == true ]]; then
    echo "🛑 Stopping Ollama after corpus preprocessing..."
    sudo docker stop ollama
fi

# --- Step 2: Start RAG Service ---
echo ""
echo "⚠️  RAG service requires minimum 8GB VRAM."
echo "For hardware/image requirements, visit: https://hub.docker.com/r/ollama/ollama"
read -p "Do you want to start the RAG service now? (y/n): " start_service
if [[ "$start_service" == "y" ]]; then
    echo "Starting Ollama and RAG services..."
    sudo docker start ollama || echo "Ollama container not found or already running."
    pull_model_if_missing "nomic-embed-text"
    pull_model_if_missing "zephyr"
    sudo docker network create rag-app-network || echo "Network already exists."
    sudo docker compose -f rag-app-compose.yml up -d
    echo "✅ RAG services are running."
    echo "Access the chatbot at: http://localhost:7860/"
else
    echo "❌ RAG service was not started. Exiting setup."
    exit 0
fi

# --- Step 3: Optional Shutdown ---
echo ""
read -p "Do you want to shut down the RAG system now? (y/n): " shutdown_now
if [[ "$shutdown_now" == "y" ]]; then
    echo "Shutting down services..."
    sudo docker compose -f rag-app-compose.yml down
    sudo docker stop ollama
    sudo docker network rm rag-app-network
    echo "✅ Shutdown complete."
else
    echo "RAG system is still running."
    echo "You can shut it down later using:"
    echo "  sudo docker compose -f rag-app-compose.yml down"
    echo "  sudo docker stop ollama"
    echo "  sudo docker network rm rag-app-network"
fi

echo "===== Done ====="
