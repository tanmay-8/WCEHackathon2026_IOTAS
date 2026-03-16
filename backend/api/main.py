from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, chat, auth, memory, documents
from config.settings import settings

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate settings on startup
@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    settings.validate()
    print(f"🚀 {settings.API_TITLE} v{settings.API_VERSION} starting...")

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(documents.router, tags=["documents"])
app.include_router(chat.router, tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.API_TITLE}",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)