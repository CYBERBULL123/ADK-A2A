# ADK & A2A Project - Deployment Guide

This directory contains deployment configurations for the ADK & A2A learning project, supporting both development and production environments.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Environment variables configured (see `.env.example`)

### Deploy Development Environment
```bash
cd deployment
./deploy.sh setup  # First time setup
./deploy.sh dev    # Deploy development environment
```

### Deploy Production Environment
```bash
./deploy.sh prod   # Deploy with Nginx reverse proxy
```

## 📁 Files Overview

### `Dockerfile`
Multi-stage Docker image for the ADK application:
- Based on Python 3.9 slim
- Installs system dependencies and Python packages
- Exposes ports for agents and dashboard
- Includes health checks

### `docker-compose.yml`
Orchestrates multiple services:
- **adk-dashboard**: Main Streamlit application
- **weather-agent**: Weather information agent
- **news-agent**: News information agent
- **redis**: Optional state management
- **nginx**: Production reverse proxy

### `nginx.conf`
Production-ready Nginx configuration:
- Reverse proxy for all services
- Rate limiting for security
- WebSocket support for Streamlit
- Health check endpoints

### `deploy.sh`
Deployment automation script:
- Environment setup and validation
- Docker image building
- Service deployment
- Health checks and monitoring

## 🔧 Configuration

### Environment Variables
Required environment variables (set in `.env`):
```bash
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ENVIRONMENT=development
```

### Port Mapping
- **8501**: Streamlit dashboard
- **8000**: Coordinator agent
- **8001**: Weather agent
- **8002**: News agent
- **6379**: Redis (optional)
- **80/443**: Nginx proxy (production)

## 🏗️ Deployment Options

### Development Deployment
```bash
# Build and start development services
./deploy.sh dev

# View logs
./deploy.sh logs

# Health check
./deploy.sh health
```

Services available at:
- Dashboard: http://localhost:8501
- Coordinator: http://localhost:8000
- Weather Agent: http://localhost:8001
- News Agent: http://localhost:8002

### Production Deployment
```bash
# Deploy with Nginx proxy
./deploy.sh prod
```

Services available at:
- Application: http://localhost
- API endpoints: http://localhost/api/*

### Service Management
```bash
# Stop all services
./deploy.sh stop

# Clean up resources
./deploy.sh cleanup

# Rebuild images
./deploy.sh build
```

## 🔍 Monitoring & Debugging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f adk-dashboard
```

### Health Checks
```bash
# Automated health check
./deploy.sh health

# Manual checks
curl http://localhost:8501/health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### Container Status
```bash
# View running containers
docker-compose ps

# Container resource usage
docker stats
```

## 🔒 Security Considerations

### Development
- Services exposed on localhost only
- Basic rate limiting via Nginx
- Environment variables in .env file

### Production
- Nginx reverse proxy with rate limiting
- SSL/TLS termination (configure certificates)
- Environment variable security
- Network isolation between services

### Best Practices
1. Use strong API keys and rotate regularly
2. Enable SSL/TLS for production deployments
3. Configure firewall rules for exposed ports
4. Monitor logs for security events
5. Regular Docker image updates

## 🚀 Cloud Deployment

### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t adk-project .
docker tag adk-project:latest <account>.dkr.ecr.us-east-1.amazonaws.com/adk-project:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/adk-project:latest
```

### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/adk-project
gcloud run deploy --image gcr.io/PROJECT-ID/adk-project --platform managed
```

### Azure Container Instances
```bash
# Create resource group and deploy
az group create --name adk-rg --location eastus
az container create --resource-group adk-rg --name adk-project --image your-registry/adk-project:latest
```

## 📊 Scaling & Performance

### Horizontal Scaling
- Multiple agent instances behind load balancer
- Redis for shared state management
- Database for persistent storage

### Vertical Scaling
- Adjust Docker resource limits
- Optimize Python application performance
- Monitor memory and CPU usage

### Load Balancing
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  weather-agent:
    deploy:
      replicas: 3
  news-agent:
    deploy:
      replicas: 2
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8501
   # Stop conflicting services
   ```

2. **Environment Variables Not Set**
   ```bash
   # Verify .env file exists and has correct values
   cat ../.env
   ```

3. **Docker Build Failures**
   ```bash
   # Clean build cache
   docker system prune -a
   # Rebuild without cache
   docker-compose build --no-cache
   ```

4. **Service Health Check Failures**
   ```bash
   # Check service logs
   docker-compose logs service-name
   # Restart specific service
   docker-compose restart service-name
   ```

### Performance Tuning
1. Adjust Docker memory limits
2. Optimize Python code and dependencies
3. Use connection pooling for external APIs
4. Implement caching strategies
5. Monitor and profile application performance

## 📝 Maintenance

### Regular Tasks
- Update Docker images regularly
- Monitor log file sizes
- Check for security updates
- Backup configuration and data
- Test disaster recovery procedures

### Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
./deploy.sh cleanup
./deploy.sh dev
```
