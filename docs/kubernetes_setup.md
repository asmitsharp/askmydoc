# AskMyDocs Kubernetes Setup Guide

This guide details how I deployed the AskMyDocs RAG system on Kubernetes. It covers the complete process from setting up the local cluster to deploying the databases (Postgres, Qdrant, Redis) and spinning up the API layer.

## Prerequisites
Before you begin, ensure you have the following installed:
- Docker
- [kind](https://kind.sigs.k8s.io/) (Kubernetes IN Docker)
- `kubectl`

## Step 1: Create the Cluster

We use `kind` with a custom config that **maps host ports 80 and 443** into the cluster node. This is required for the NGINX Ingress Controller to be reachable directly at `http://localhost` without needing a separate `port-forward`.

Create the cluster config file:

```bash
cat <<'EOF' > kind-config.yml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      - containerPort: 443
        hostPort: 443
        protocol: TCP
EOF
```

Then create the cluster:

```bash
kind create cluster --name askmydocs --config kind-config.yml
```

To enable the Horizontal Pod Autoscaler (HPA) to work, we also need to install the `metrics-server`. Because `kind` uses self-signed certificates, we must patch the deployment with `--kubelet-insecure-tls`.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'
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

And that's it for the core deployment! Steps 7 and 8 below complete the production setup with **Ingress routing** and **dynamic observability**.

---

## Step 7: Ingress — Production Network Routing

With the port-mapped cluster from Step 1, you no longer need `kubectl port-forward` for the API. An NGINX Ingress Controller handles all external traffic and routes it to the right internal Service.

### 7a. Install the NGINX Ingress Controller

The `kind`-specific manifest pre-configures the NGINX pods with the correct node tolerations and labels for `kind` clusters:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
```

Wait until the controller pod is fully ready (takes ~60s):

```bash
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

### 7b. Apply the updated Service and Ingress manifests

The Service has been updated from `NodePort` to `ClusterIP` (since Ingress now handles external traffic) and has named ports required by the Ingress and ServiceMonitor:

```bash
kubectl apply -f k8s/api/service.yml
kubectl apply -f k8s/ingress/ingress.yml
```

### 7c. Verify routing

You can now hit the API directly on port 80 — no `port-forward` needed:

```bash
curl http://localhost/api/health
# Expected: {"llm":true,"status":"ok","vector_db":true}

curl -X POST http://localhost/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AskMyDocs?"}'
```

> **How the path rewrite works:** The Ingress maps `/api/*` to the backend and strips the `/api` prefix before forwarding. So `GET /api/health` becomes `GET /health` by the time it reaches your Go pod — matching your existing router exactly.

---

## Step 8: Prometheus Operator & ServiceMonitors — Dynamic Observability

In Docker Compose, Prometheus had a static target (`api:8080`). In Kubernetes, pods are ephemeral and get new IPs on every restart or scale event. The **Prometheus Operator** solves this by watching `ServiceMonitor` resources and automatically updating Prometheus's scrape targets whenever pods change.

### 8a. Install Helm

Helm is a package manager for Kubernetes — it installs the entire Prometheus stack (Prometheus, Grafana, Alertmanager, Operator, and 40+ CRDs) with a single command:

```bash
brew install helm
helm version   # verify
```

### 8b. Add the Prometheus community chart repository

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### 8c. Install `kube-prometheus-stack`

```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=askmydocs \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

> `serviceMonitorSelectorNilUsesHelmValues=false` is critical — without it, Prometheus ignores ServiceMonitors that aren't in the `monitoring` namespace.

Watch all monitoring pods come up (takes 2–3 minutes):

```bash
kubectl get pods -n monitoring -w
```

Wait until you see all pods in `Running` state before continuing.

### 8d. Apply the ServiceMonitor

```bash
kubectl apply -f k8s/api/servicemonitor.yml
```

Verify it was registered:

```bash
kubectl get servicemonitor -n monitoring
# Expected: askmydocs-api-monitor
```

### 8e. Verify Prometheus is scraping the API

Port-forward to the Prometheus UI:

```bash
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
```

Open **http://localhost:9090/targets** — you should see `askmydocs/askmydocs-api-monitor` with state `UP`. It may take up to 30 seconds after applying the ServiceMonitor.

### 8f. Access Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3001:80
```

Open **http://localhost:3001**

- **Username:** `admin`
- **Password:** `askmydocs`

To restore your custom dashboards:
1. Go to **Dashboards → Import**
2. Upload the JSON files from `docker/grafana/`
3. Select `Prometheus` as the data source (auto-configured by Helm)

---

## Full Setup Verification Checklist

Run these in order to confirm everything is healthy end-to-end:

```bash
# Core cluster
kubectl get pods -n askmydocs

# Ingress controller
kubectl get pods -n ingress-nginx

# API reachable via Ingress (no port-forward needed)
curl http://localhost/api/health

# Monitoring stack
kubectl get pods -n monitoring

# ServiceMonitor registered
kubectl get servicemonitor -n monitoring

# Prometheus targets UI
# Open http://localhost:9090/targets after:
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
```
