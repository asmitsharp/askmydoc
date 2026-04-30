# AskMyDocs Kubernetes Setup Guide

This guide details how I deployed the AskMyDocs RAG system on Kubernetes. It covers the complete process from setting up the local cluster to deploying the databases (Postgres, Qdrant, Redis) and spinning up the API layer.

## Prerequisites
Before you begin, ensure you have the following installed:
- Docker
- [kind](https://kind.sigs.k8s.io/) (Kubernetes IN Docker)
- `kubectl`

## Step 1: Create the Cluster
I used `kind` to spin up a local Kubernetes cluster named `askmydocs`.

```bash
kind create cluster --name askmydocs
```

## Step 2: Set Up Secrets and Configurations

Because AskMyDocs relies on several API keys and external connections, I created a Kubernetes Secret to keep them secure.

First, ensure your `.env` file is ready, or use the pre-configured `askmydocs-secrets.yaml` inside the `k8s/secrets` directory. Then, apply the namespace, ConfigMap, and Secrets:

```bash
# Create the namespace
kubectl apply -f k8s/namespace.yml

# Apply configurations and secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets/askmydocs-secrets.yaml
```

## Step 3: Deploy Databases & Cache

Our system requires **Qdrant** (Vector DB), **PostgreSQL** (Relational/BM25 DB), and **Redis** (Message queue for asynchronous ingestion).

I set up PersistentVolumeClaims (PVCs), StatefulSets, and Services for Qdrant and Postgres, and a simple Deployment for Redis.

Run the following commands to spin them up:

```bash
# 1. Apply Persistent Volume Claims
kubectl apply -f k8s/qdrant/pvc.yml
kubectl apply -f k8s/postgres/pvc.yml

# 2. Apply StatefulSets & Deployments
kubectl apply -f k8s/qdrant/statefulset.yml
kubectl apply -f k8s/postgres/statefulset.yml
kubectl apply -f k8s/redis/deployment.yml

# 3. Apply Services
kubectl apply -f k8s/qdrant/service.yml
kubectl apply -f k8s/postgres/service.yml
kubectl apply -f k8s/redis/service.yml
```

You can check if they are running by executing:
```bash
kubectl get pods -n askmydocs
```
Wait until `qdrant-0`, `postgres-0`, and the `redis` pods are in the `Running` state.

## Step 4: Build and Load the API Image

Since we are running this in a local `kind` cluster, the cluster needs access to our AskMyDocs API Docker image. I configured the deployment with `imagePullPolicy: Never` so that it doesn't try to pull from the internet.

Build the image locally and load it into the cluster:

```bash
# Build the image locally
docker build -t askmydocs/askmydocs-api:v0.7.0 -f docker/Dockerfile .

# Load the image into the kind cluster
kind load docker-image askmydocs/askmydocs-api:v0.7.0 --name askmydocs
```

## Step 5: Deploy the API and HPA

Now that the image is available and our databases are running, I deployed the API layer, its Service, and the Horizontal Pod Autoscaler (HPA) to automatically scale up if traffic increases.

```bash
kubectl apply -f k8s/api/deployment.yml
kubectl apply -f k8s/api/service.yml
kubectl apply -f k8s/api/hpa.yml
```

If you ever need to forcefully restart the deployment to pick up a new image or config, run:
```bash
kubectl rollout restart deployment askmydocs-api -n askmydocs
```

## Step 6: Testing the Deployment

To test the system locally, I port-forwarded the API service to my local machine:

```bash
kubectl port-forward svc/askmydocs-api-service 8080:80 -n askmydocs
```

*(Keep that terminal tab open and open a new one for testing)*

### 1. Health Check
```bash
curl http://localhost:8080/health
# Expected Output: {"llm":true,"status":"ok","vector_db":true}
```

### 2. Ingest a Document
```bash
echo "AskMyDocs is an advanced AI-powered documentation search assistant built on top of Kubernetes and Qdrant." > test-doc.txt
curl -X POST -F "file=@test-doc.txt" http://localhost:8080/ingest
```

### 3. Query the RAG Pipeline
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AskMyDocs?"}'
```

And that's it! The whole distributed setup is running seamlessly in Kubernetes!
