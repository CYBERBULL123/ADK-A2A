#!/bin/bash

# ADK & A2A Project Deployment Script
# This script helps deploy the project to various environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Setup environment file
setup_environment() {
    if [ ! -f "../.env" ]; then
        print_warning ".env file not found. Creating from template..."
        cp ../.env.example ../.env
        print_warning "Please edit .env file with your API keys before proceeding"
        read -p "Press enter to continue after setting up .env file..."
    else
        print_success ".env file found"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    docker-compose build
    print_success "Docker images built successfully"
}

# Deploy development environment
deploy_development() {
    print_status "Deploying development environment..."
    docker-compose up -d adk-dashboard weather-agent news-agent redis
    print_success "Development environment deployed"
    print_status "Dashboard available at: http://localhost:8501"
    print_status "Agent coordinator at: http://localhost:8000"
    print_status "Weather agent at: http://localhost:8001"
    print_status "News agent at: http://localhost:8002"
}

# Deploy production environment
deploy_production() {
    print_status "Deploying production environment with Nginx..."
    docker-compose --profile production up -d
    print_success "Production environment deployed"
    print_status "Application available at: http://localhost"
    print_status "Nginx proxy handling requests"
}

# Show logs
show_logs() {
    print_status "Showing logs for all services..."
    docker-compose logs -f
}

# Stop all services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Cleanup Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Health check
health_check() {
    print_status "Performing health check..."
    
    services=("adk-dashboard:8501" "weather-agent:8001" "news-agent:8002")
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if curl -f -s http://localhost:$port/health > /dev/null 2>&1; then
            print_success "$name is healthy"
        else
            print_warning "$name health check failed or service not ready"
        fi
    done
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev         Deploy development environment"
    echo "  prod        Deploy production environment with Nginx"
    echo "  build       Build Docker images"
    echo "  logs        Show logs from all services"
    echo "  health      Perform health check on services"
    echo "  stop        Stop all services"
    echo "  cleanup     Stop services and cleanup Docker resources"
    echo "  setup       Setup environment and dependencies"
    echo ""
    echo "Examples:"
    echo "  $0 setup       # First time setup"
    echo "  $0 dev         # Deploy for development"
    echo "  $0 prod        # Deploy for production"
    echo "  $0 logs        # View logs"
    echo "  $0 stop        # Stop services"
}

# Main script logic
main() {
    case "${1:-}" in
        "setup")
            check_dependencies
            setup_environment
            build_images
            ;;
        "dev")
            check_dependencies
            setup_environment
            build_images
            deploy_development
            sleep 5
            health_check
            ;;
        "prod")
            check_dependencies
            setup_environment
            build_images
            deploy_production
            sleep 5
            health_check
            ;;
        "build")
            check_dependencies
            build_images
            ;;
        "logs")
            show_logs
            ;;
        "health")
            health_check
            ;;
        "stop")
            stop_services
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        "")
            print_error "No command specified"
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
