#!/bin/bash

# professional deployment script

echo "starting deployment process..."

# build docker image
echo "building docker image..."
docker build -f docker/Dockerfile -t intelligent-doc-system:latest .

# tag for registry
echo "tagging image..."
docker tag intelligent-doc-system:latest your-registry/intelligent-doc-system:latest

# push to registry
echo "pushing to registry..."
docker push your-registry/intelligent-doc-system:latest

# deploy to cloud
echo "deploying to cloud..."
# add your cloud deployment commands here

echo "deployment completed successfully"
