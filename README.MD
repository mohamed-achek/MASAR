# 🎓 Masar — مسار

<div align="center">

**AI-Powered Education & Career Guidance Platform**

[![OpenShift](https://img.shields.io/badge/OpenShift-Ready-red?logo=redhatopenshift)](https://masar-frontend-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.11-green?logo=python)](requirements.txt)
[![React](https://img.shields.io/badge/React-18-blue?logo=react)](frontend/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

[🚀 Live Demo](https://masar-frontend-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com) • 
[📖 Documentation](OPENSHIFT_README.md) • 
[📋 OpenShift Guide](openshift/README.md)

</div>

---

## 🌟 Overview

**Masar** (Arabic for "path" or "journey") is an AI-powered Retrieval-Augmented Generation (RAG) application designed to help Tunisian students and job seekers navigate educational and career opportunities. The platform uses semantic search and large language models to provide intelligent, context-aware responses about university programs, courses, and career guidance.


### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **RAG-Powered Q&A** | Semantic search with LLM-generated responses |
| 🔐 **User Authentication** | JWT-based secure login and registration |
| 🎓 **Education Pathways** | University program recommendations |
| 📊 **Skill Gap Analysis** | Career-aligned learning suggestions |
| 🌍 **Multilingual** | English, French, and Arabic support |
| 📈 **Market Insights** | Real-time job market trends |
| 🐳 **Container-Ready** | Docker Compose & OpenShift deployment |
| ⚡ **Auto-Scaling** | HPA-based horizontal scaling |


## 🌐 Live Deployment

| Component | URL | Status |
|-----------|-----|--------|
| **Frontend** | https://masar-frontend-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com | ✅ Running |
| **Backend API** | https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api | ✅ Running |

> ⚠️ **Note:** RAG/semantic search is disabled on OpenShift due to Developer Sandbox memory limits (7Gi). Full RAG functionality is available in local Docker Compose deployment with Ollama.

---

## 📁 Project Structure

```
masar/
├── api_server.py              # FastAPI backend application
├── database.py                # SQLAlchemy database models
├── docker-compose.yml         # Local development setup
├── Dockerfile.backend         # Backend container image
├── Dockerfile.frontend        # Frontend container image
├── requirements.txt           # Python dependencies (RAG)
├── api_requirements.txt       # Python dependencies (API)
│
├── app/                       # Application modules
│   ├── input_handler.py       # Query preprocessing
│   ├── intent_ner.py          # Intent recognition
│   ├── preprocessing.py       # Text preprocessing
│   └── retriever.py           # Document retrieval
│
├── pipeline/                  # RAG pipeline components
│   ├── embedding.py           # Vector embeddings
│   ├── retrieve.py            # FAISS retrieval
│   ├── rag_new.py             # RAG orchestration
│   └── table_processor.py     # PDF table extraction
│
├── frontend/                  # React application
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   ├── components/
│   │   └── store/
│   └── package.json
│
├── openshift/                 # OpenShift manifests
│   ├── README.md              # Deployment guide
│   ├── deploy.sh              # Automated deployment
│   ├── 01-postgresql-*.yaml   # Database deployment
│   ├── 02-backend-*.yaml      # Backend deployment
│   ├── 03-frontend-*.yaml     # Frontend deployment
│   ├── 04-persistent-*.yaml   # Storage (PVCs)
│   ├── 05-secrets.yaml        # Credentials
│   ├── 06-configmaps.yaml     # Configuration
│   ├── 07-services.yaml       # Internal networking
│   ├── 08-routes.yaml         # External access
│   └── 09-hpa.yaml            # Auto-scaling
│
├── data/                      # RAG data directory
│   ├── raw/                   # Source documents
│   ├── processed/             # Preprocessed data
│   └── final/                 # Production indices
│       ├── chunks/            # Document chunks
│       ├── embeddings/        # Vector embeddings
│       └── indices/           # FAISS indices
│
└── docs/                      # Documentation
    └── IT460_Final_Report.tex # LaTeX project report
```

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.11 | Runtime |
| FastAPI | REST API framework |
| SQLAlchemy | Async ORM |
| PostgreSQL 15 | Database |
| FAISS | Vector similarity search |
| sentence-transformers | Text embeddings |
| Ollama | LLM inference |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 18 | UI framework |
| Vite | Build tool |
| Tailwind CSS | Styling |
| Zustand | State management |
| Nginx | Static serving & proxy |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Docker Compose | Local orchestration |
| OpenShift 4.14+ | Production orchestration |
| HAProxy | Ingress routing |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [OPENSHIFT_README.md](OPENSHIFT_README.md) | Complete architecture and deployment guide |
| [openshift/README.md](openshift/README.md) | OpenShift-specific setup instructions |
---

## 🎯 IT460 Course Requirements

This project satisfies the IT460 Multi-Container Application requirements:

| Requirement | Implementation |
|-------------|----------------|
| ✅ Multi-container | 3 microservices (Frontend, Backend, PostgreSQL) |
| ✅ DeploymentConfigs | Frontend, PostgreSQL deployments |
| ✅ Services | ClusterIP services for all components |
| ✅ Routes | HTTPS with TLS edge termination |
| ✅ PersistentVolumeClaims | PostgreSQL data persistence |
| ✅ ConfigMaps | Nginx configuration |
| ✅ Secrets | Database credentials, JWT secret |
| ✅ HorizontalPodAutoscaler | CPU/Memory-based scaling |
| ✅ Rolling Updates | Zero-downtime deployments |
| ✅ Health Checks | Liveness/Readiness probes |

---

## 🔧 API Reference

### Authentication

```bash
# Login
curl -X POST https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@masar.tn", "password": "admin123"}'

# Response
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "user": {"email": "admin@masar.tn", "name": "Admin User"}
}
```

### Health Check

```bash
curl https://masar-backend-api-m-achek-dev.apps.rm1.0a51.p1.openshiftapps.com/api/health

# Response
{
  "status": "healthy",
  "timestamp": "2026-01-17T12:00:00",
  "hostname": "masar-backend-deploy-xxx",
  "version": "1.0.0"
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **IT460 Course** - Container orchestration concepts and guidance
- **Red Hat OpenShift** - Developer Sandbox platform
- **Hugging Face** - Transformer models and embeddings
- **Ollama** - Local LLM inference

---

<div align="center">

**Built with ❤️ for Tunisian Education**

[⬆ Back to Top](#-masar--مسار)

</div>

