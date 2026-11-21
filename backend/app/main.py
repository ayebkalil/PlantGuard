from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from backend.app.api.routes import router
from backend.app.config import CORS_ORIGINS, MAX_BATCH_FILES, MAX_UPLOAD_BYTES


class LimitInferenceBody:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        path = scope["path"].rstrip("/")
        if path not in {"/predict", "/predict/batch"} and not path.startswith("/vision/"):
            await self.app(scope, receive, send)
            return

        count = MAX_BATCH_FILES if path == "/predict/batch" else 1
        limit = count * MAX_UPLOAD_BYTES + 128 * 1024  # multipart headers and framing
        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and int(length) > limit:
            await JSONResponse({"detail": "Request exceeds the upload limit"}, status_code=413)(scope, receive, send)
            return
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                raise HTTPException(413, "Request exceeds the upload limit")
            return message

        await self.app(scope, bounded_receive, send)

app = FastAPI(
    title="PlantGuard API",
    description="Plant disease detection and MLOps platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(LimitInferenceBody)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"message": "PlantGuard backend is running"}
