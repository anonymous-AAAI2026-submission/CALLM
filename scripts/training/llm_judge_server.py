import argparse
import logging
import traceback
from typing import List

import uvicorn
import yaml
from fastapi import FastAPI
from llm_judge import BaseLLMJudge, LLMJudgeConfig
from pydantic import BaseModel

# ---------- Setup Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> LLMJudgeConfig:
    with open(path) as f:
        raw_config = yaml.safe_load(f)
    return LLMJudgeConfig(**raw_config)


def create_app(judge: BaseLLMJudge) -> FastAPI:
    logger.info("🔧 Creating FastAPI app...")
    app = FastAPI()

    # ---------- Schema Definitions ----------
    class JudgeItem(BaseModel):
        rules: str
        reasoning: str
        final_answer: str

    class JudgeBatchRequest(BaseModel):
        items: List[JudgeItem]

    class JudgeResponseItem(BaseModel):
        score: float
        response: str

    # ---------- Batch Evaluation ----------
    @app.post("/evaluate_batch", response_model=List[JudgeResponseItem])
    async def evaluate_batch(req: JudgeBatchRequest):
        try:
            logger.info(f"📥 Received batch request with {len(req.items)} items")
            reasonings = [item.reasoning for item in req.items]
            final_answers = [item.final_answer for item in req.items]
            rules = req.items[0].rules  # assume same rules for batch

            scores, responses = judge.evaluate_reasoning_batch(rules, reasonings, final_answers)
            logger.info("✅ Batch evaluation complete")
            return [{"score": s, "response": r} for s, r in zip(scores, responses)]

        except Exception as e:
            logger.error("❌ ERROR in /evaluate_batch")
            logger.error(traceback.format_exc())
            raise e

    @app.get("/health/")
    async def health_check():
        return {"status": "ok"}

    # ---------- Single Evaluation ----------
    @app.post("/evaluate", response_model=JudgeResponseItem)
    async def evaluate(req: JudgeItem):
        try:
            logger.info("📥 Received single evaluation request")
            score, response = judge.evaluate_reasoning(req.rules, req.reasoning, req.final_answer)
            logger.info("✅ Single evaluation complete")
            return {"score": score, "response": response}

        except Exception as e:
            logger.error("❌ ERROR in /evaluate")
            logger.error(traceback.format_exc())
            raise e

    logger.info("🚀 FastAPI app created successfully")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        logger.info("📄 Loading config...")
        config = load_config(args.config)

        logger.info("🤖 Instantiating judge...")
        judge = BaseLLMJudge(config)

        logger.info("🧱 Creating app...")
        app = create_app(judge)

        logger.info(f"🚀 Starting server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    except Exception as e:
        logger.error("❌ Failed to start server:")
        logger.error(traceback.format_exc())
        exit(1)
