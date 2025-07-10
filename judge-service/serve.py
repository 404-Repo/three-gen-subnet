import argparse
import sys

import uvicorn
from application.api import api_router
from fastapi import FastAPI
from loguru import logger


logger.remove()
logger.add(sink=sys.stdout, level="DEBUG", filter=lambda record: record["level"].no < 40)  # Everything below ERROR
logger.add(sink=sys.stderr, level="ERROR")  # ERROR and above


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10007)
    args, _ = parser.parse_known_args()
    return args


app = FastAPI()
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port)
