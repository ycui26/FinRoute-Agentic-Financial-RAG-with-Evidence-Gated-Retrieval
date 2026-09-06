#!/usr/bin/env python
"""Small local demo that prints retrieval diagnostics as JSON."""

import argparse
import json

from finroute.config import FinRouteConfig
from finroute.pipeline import FinRoutePipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    config = FinRouteConfig.from_yaml(args.config)
    pipeline = FinRoutePipeline.from_artifacts(args.artifacts, config)
    print(json.dumps(pipeline.ask(args.question, args.generate), indent=2, default=str))


if __name__ == "__main__":
    main()
