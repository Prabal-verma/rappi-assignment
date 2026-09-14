"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api import routes_agent, routes_data
from app.config import settings
from app.db.models import Product
from app.db.seed import seed_all
from app.db.session import SessionLocal, init_db

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("purchasing-agent")

app = FastAPI(
    title="AI Purchasing Agent",
    version="1.0.0",
    description=(
        "An agent that reviews purchasing situations, gathers the evidence it needs, decides, "
        "executes within hard constraints, and validates the result against what it predicted."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_agent.router, prefix="/api", tags=["agent"])
app.include_router(routes_data.router, prefix="/api/data", tags=["data"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    with SessionLocal() as session:
        if session.execute(select(Product).limit(1)).scalar_one_or_none() is None:
            logger.info("Empty database — seeding scenario data.")
            seed_all(session)
            session.commit()
    logger.info(
        "Ready. LLM provider=%s model=%s live=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.llm_is_live,
    )


@app.get("/")
def root() -> dict:
    return {
        "name": "AI Purchasing Agent",
        "docs": "/docs",
        "health": "/api/health",
        "scenarios": "/api/scenarios",
    }
