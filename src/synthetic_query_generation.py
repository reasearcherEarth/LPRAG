import json
import logging
import re
from typing import List

import yaml

from src.concurrent_utils import load_checkpoint, process_items_concurrently
from src.llms import build_llm
from src.prompts import prompt_build_syn_query_2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QUERY_PATTERN = re.compile(r"\[(.*?\?)\]")


def build_chunks_prompt(chunk_list: List[str]) -> str:
    lines = []
    for idx_chunk, chunk in enumerate(chunk_list):
        lines.append(f"[Chunk {idx_chunk}]")
        lines.append(chunk)
    joined = "\n".join(lines)
    return f"[The Start bag of Chunks]\n{joined}\n[The End bag of Chunks]"


def parse_queries(response: str) -> List[List[str]]:
    if not response:
        return []
    queries = QUERY_PATTERN.findall(response)
    return [queries[i : i + 2] for i in range(0, len(queries), 2)]


def generate_queries_for_document(
    llm, chunk_list: List[str], prompt: str = prompt_build_syn_query_2
) -> List[List[str]]:
    user_prompt = build_chunks_prompt(chunk_list)
    response, _tokens = llm.generate(prompt, user_prompt, n=1)
    return parse_queries(response)


def main():
    logger.info("generating queries...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(config["path_chunks"], "r", encoding="utf-8") as f:
        chunks_by_doc = json.load(f)

    llm = build_llm(config)
    checkpoint_path = config["destiny_path_syn_query"]
    max_workers = config.get("max_workers", 8)

    results = load_checkpoint(checkpoint_path)
    items = [
        (doc_idx, chunk_list)
        for doc_idx, chunk_list in chunks_by_doc.items()
        if str(doc_idx) not in results
    ]

    def worker(chunk_list: List[str]) -> List[List[str]]:
        return generate_queries_for_document(llm, chunk_list)

    process_items_concurrently(items, worker, results, checkpoint_path, max_workers=max_workers)

    logger.info("Queries done! number of documents process: %d", len(results))


if __name__ == "__main__":
    main()