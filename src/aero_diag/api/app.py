"""FastAPI 应用工厂——控制面 API 入口。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aero_diag.api.routes import assets, reviews, runs, tasks
from aero_diag.infrastructure.config import config


def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(
        title="Aero-Engine Damage Detection System",
        description="航空发动机损伤诊断智能体系统 API — 控制面",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(assets.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")
    app.include_router(reviews.router, prefix="/api/v1")

    @app.get("/health")
    def health_check() -> dict:
        return {
            "status": "healthy",
            "version": "0.1.0",
            "deployment_mode": config.deployment_mode,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "aero_diag.api.app:app",
        host=config.host,
        port=config.port,
        reload=config.deployment_mode == "development",
        log_level=config.log_level.lower(),
    )
