import ssl
import urllib3
import os

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
from app.core.config import settings
from app.core.logging import logger
from app.api.endpoints import router
from app.models.response import ErrorResponse

# create app instance
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
)

# add cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# add request timing middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    
    return response

# exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            detail=exc.detail,
            status_code=exc.status_code
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            detail="an unexpected error occurred",
            status_code=500
        ).dict()
    )

# include routers
app.include_router(router, prefix="/api/v1")

# root endpoint
@app.get("/")
async def root():
    return {
        "service": "intelligent doc query system",
        "version": settings.api_version,
        "status": "operational"
    }

# startup event
@app.on_event("startup")
async def startup_event():
    logger.info("intelligent doc query system starting up")
    logger.info("SSL verification disabled for model downloads")

# shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("intelligent doc query system shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug
    )