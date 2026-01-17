# Masar RAG Application - OpenShift Deployment

## IT460: Multi-Container Application Development using OpenShift

This directory contains all OpenShift manifests and configuration files needed to deploy the Masar RAG application.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Architecture Overview](#-architecture-overview)
3. [Prerequisites](#-prerequisites)
4. [File Structure](#-file-structure)
5. [Deployment Guide](#-deployment-guide)
6. [Accessing the Application](#-accessing-the-application)
7. [Operations Guide](#-operations-guide)
8. [Troubleshooting](#-troubleshooting)
9. [IT460 Requirements Mapping](#-it460-requirements-mapping)

---

## 🚀 Quick Start

```bash
# 1. Login to OpenShift
oc login https://api.rm1.0a51.p1.openshiftapps.com:6443

# 2. Create/switch to project
oc project m-achek-dev

# 3. Deploy everything
./deploy.sh

# 4. Access the application
# Frontend: https://masar-frontend-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com
# Backend API: https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api
```



## 🏗 Architecture Overview

### Components

| Component | Type | Replicas | Port | Description |
|-----------|------|----------|------|-------------|
| **Frontend** | DeploymentConfig | 2 | 8080 | React SPA served by nginx |
| **Backend** | Deployment | 1 | 8000 | FastAPI REST API |
| **PostgreSQL** | DeploymentConfig | 1 | 5432 | Database for user management |

---

## 📝 Prerequisites

### Required Tools

```bash
# OpenShift CLI
oc version
# Client Version: 4.14+

# Optional: jq for JSON parsing
jq --version
```

### OpenShift Access

- Access to an OpenShift 4.x cluster
- Permission to create resources in your namespace
- Access to the internal container registry

### Verify Access

```bash
# Check you're logged in
oc whoami

# Check current project
oc project

# Check available storage classes
oc get storageclass
```

---

## 📁 File Structure

```
openshift/
├── README.md                           # This file
├── deploy.sh                           # Automated deployment script
├── kustomization.yaml                  # Kustomize configuration
├── nginx.conf                          # Frontend nginx configuration
│
├── 01-postgresql-deploymentconfig.yaml # PostgreSQL database
├── 02-backend-deploymentconfig.yaml    # Backend API (DeploymentConfig)
├── 03-frontend-deploymentconfig.yaml   # Frontend web app
├── 04-persistent-volume-claims.yaml    # Storage for PostgreSQL
├── 05-secrets.yaml                     # Database & JWT credentials
├── 06-configmaps.yaml                  # nginx configuration
├── 07-services.yaml                    # Internal networking
├── 08-routes.yaml                      # External access (HTTPS)
├── 09-hpa.yaml                         # Horizontal Pod Autoscaler
├── 10-imagestreams.yaml                # Container image references
└── 11-buildconfigs.yaml                # Build configurations
```

### Resource Descriptions

| File | Resources | Purpose |
|------|-----------|---------|
| `01-postgresql-*` | DeploymentConfig | PostgreSQL 15 database |
| `02-backend-*` | DeploymentConfig | FastAPI backend (legacy, use Deployment) |
| `03-frontend-*` | DeploymentConfig | React + nginx frontend |
| `04-persistent-*` | PVC | PostgreSQL data storage |
| `05-secrets.yaml` | Secret | DB credentials, JWT secret |
| `06-configmaps.yaml` | ConfigMap | nginx proxy configuration |
| `07-services.yaml` | Service | ClusterIP services for all components |
| `08-routes.yaml` | Route | HTTPS ingress with TLS |
| `09-hpa.yaml` | HPA | Auto-scaling configuration |
| `10-imagestreams.yaml` | ImageStream | Image references |
| `11-buildconfigs.yaml` | BuildConfig | S2I build configurations |

---

## 📦 Deployment Guide

### Method 1: Automated Deployment (Recommended)

```bash
# Navigate to openshift directory
cd openshift

# Make script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### Method 2: Manual Step-by-Step Deployment

#### Step 1: Create Secrets

```bash
# Create database credentials secret
oc create secret generic masar-db-credentials \
  --from-literal=database-url="postgresql+asyncpg://masar_user:MasarSecurePass2024!@masar-postgresql:5432/masar_db" \
  --from-literal=postgres-user="masar_user" \
  --from-literal=postgres-password="MasarSecurePass2024!" \
  --from-literal=postgres-db="masar_db"

# Create JWT secret
oc create secret generic masar-jwt-secret \
  --from-literal=jwt-secret="$(openssl rand -base64 32)"
```

#### Step 2: Create ConfigMaps

```bash
oc apply -f 06-configmaps.yaml
```

#### Step 3: Create PVCs

```bash
oc apply -f 04-persistent-volume-claims.yaml

# Wait for PVC to be bound
oc get pvc -w
```

#### Step 4: Create Services

```bash
oc apply -f 07-services.yaml
```

#### Step 5: Deploy PostgreSQL

```bash
oc apply -f 01-postgresql-deploymentconfig.yaml

# Wait for PostgreSQL to be ready
oc rollout status dc/masar-postgresql
```

#### Step 6: Build and Deploy Backend

```bash
# Create ImageStream
oc apply -f 10-imagestreams.yaml

# Create BuildConfig
oc apply -f 11-buildconfigs.yaml

# Start build from source
cd ..  # Go to project root
oc start-build masar-backend --from-dir=. --follow

# Create backend Deployment (not DeploymentConfig for memory optimization)
cat <<EOF | oc apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: masar-backend-deploy
  labels:
    app: masar
    component: backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: masar
      component: backend
  template:
    metadata:
      labels:
        app: masar
        component: backend
    spec:
      containers:
      - name: backend
        image: image-registry.openshift-image-registry.svc:5000/m-achek-dev/masar-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: masar-db-credentials
              key: database-url
        - name: SKIP_ML_MODELS
          value: "true"
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: masar-jwt-secret
              key: jwt-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "500m"
EOF
```

#### Step 7: Build and Deploy Frontend

```bash
# Build frontend
oc start-build masar-frontend --from-dir=. --follow

# Deploy frontend
oc apply -f 03-frontend-deploymentconfig.yaml

# Wait for rollout
oc rollout status dc/masar-frontend
```

#### Step 8: Create Routes

```bash
oc apply -f 08-routes.yaml
```

#### Step 9: Verify Deployment

```bash
# Check all pods are running
oc get pods

# Check services
oc get svc

# Check routes
oc get routes

# Test API health
curl -s https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api/health | jq .
```

---

## 🌐 Accessing the Application

### URLs

| Component | URL |
|-----------|-----|
| **Frontend** | https://masar-frontend-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com |
| **Backend API** | https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api |

### Default Credentials

```
Email: admin@masar.tn
Password: admin123
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/auth/login` | POST | User login |
| `/api/auth/signup` | POST | User registration |
| `/api/auth/me` | GET | Get current user |

### Test Login via CLI

```bash
# Test login
curl -s -X POST https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@masar.tn", "password": "admin123"}' | jq .

# Expected response:
# {
#   "access_token": "eyJ...",
#   "token_type": "bearer",
#   "user": {
#     "email": "admin@masar.tn",
#     "name": "Admin User"
#   }
# }
```

---

## 🔧 Operations Guide

### Scaling

```bash
# Scale frontend
oc scale dc/masar-frontend --replicas=3

# Scale backend
oc scale deployment/masar-backend-deploy --replicas=2

# Check pod distribution
oc get pods -o wide
```

### Rolling Updates

```bash
# Update frontend
oc rollout latest dc/masar-frontend
oc rollout status dc/masar-frontend

# Update backend
oc rollout restart deployment/masar-backend-deploy
oc rollout status deployment/masar-backend-deploy

# Rollback if needed
oc rollout undo dc/masar-frontend
```

### Viewing Logs

```bash
# Frontend logs
oc logs -f dc/masar-frontend

# Backend logs
oc logs -f deployment/masar-backend-deploy

# PostgreSQL logs
oc logs -f dc/masar-postgresql
```

### Database Access

```bash
# Connect to PostgreSQL
oc exec -it dc/masar-postgresql -- psql -U masar_user -d masar_db

# Run a query
oc exec -it dc/masar-postgresql -- psql -U masar_user -d masar_db -c "SELECT * FROM users;"
```

### Resource Monitoring

```bash
# Pod resource usage
oc adm top pods

# Node resource usage
oc adm top nodes

# Describe pod details
oc describe pod <pod-name>
```

---

## 🔍 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Pod `CrashLoopBackOff` | App error | `oc logs <pod>` to check |
| Pod `OOMKilled` | Memory limit exceeded | Increase limits or enable `SKIP_ML_MODELS` |
| Pod `CreateContainerConfigError` | Missing secret/configmap | `oc describe pod <pod>` |
| PVC `Pending` | Storage class issue | Check `oc get sc` |
| Route returns 503 | No healthy pods | Check pod status |

### Diagnostic Commands

```bash
# Check pod status
oc get pods

# Describe failing pod
oc describe pod <pod-name>

# Check events
oc get events --sort-by=.lastTimestamp

# Check service endpoints
oc get endpoints

# Test internal connectivity
oc debug deployment/masar-backend-deploy -- curl -s masar-postgresql:5432
```

### Reset Deployment

```bash
# Delete and recreate backend
oc delete deployment masar-backend-deploy
# Then redeploy using Step 6 above

# Restart PostgreSQL
oc rollout latest dc/masar-postgresql

# Full reset (careful!)
oc delete all -l app=masar
oc delete pvc -l app=masar
oc delete secret masar-db-credentials masar-jwt-secret
```

---

## ✅ IT460 Requirements Mapping

| Requirement | Implementation | Files |
|-------------|----------------|-------|
| **Multi-container** | 3 microservices | All deployment files |
| **DeploymentConfig** | Frontend, PostgreSQL | `01-*.yaml`, `03-*.yaml` |
| **Services** | ClusterIP for all components | `07-services.yaml` |
| **Routes** | HTTPS with TLS termination | `08-routes.yaml` |
| **PersistentVolumeClaim** | PostgreSQL data | `04-persistent-*.yaml` |
| **ConfigMaps** | nginx configuration | `06-configmaps.yaml` |
| **Secrets** | DB credentials, JWT | `05-secrets.yaml` |
| **HorizontalPodAutoscaler** | Backend auto-scaling | `09-hpa.yaml` |
| **Rolling Updates** | Zero-downtime deploys | All DeploymentConfigs |
| **Health Checks** | Liveness/Readiness probes | All deployments |

---

## 📚 Additional Resources

- [OpenShift Documentation](https://docs.openshift.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Project Main README](../README.MD)
- [Demo Script](../DEMO_SCRIPT.md)
- [Full Architecture Guide](../OPENSHIFT_README.md)

---

## ⚠️ Important Notes

1. **Memory Constraints**: The OpenShift Developer Sandbox has a 7Gi memory limit. ML models are disabled (`SKIP_ML_MODELS=true`) to allow the backend to run within these constraints.

2. **RAG Functionality**: Full RAG (semantic search) requires ~4GB RAM for ML models. Use local Podman deployment for full functionality.

3. **TLS Certificates**: Routes use edge TLS termination with OpenShift's wildcard certificate.

4. **Image Registry**: Images are stored in the internal OpenShift registry. For external registries, update ImageStreams accordingly.
