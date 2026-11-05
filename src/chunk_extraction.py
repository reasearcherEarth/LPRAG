
import json
import logging
import re
import time
from typing import Dict, List

import yaml

from src.concurrent_utils import load_checkpoint, process_items_concurrently
from src.llms import build_llm
from src.prompts import prompt_chunk_extraction
from src.utils import open_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNK_BLOCK_PATTERN = re.compile(
    r"\[Extracted Chunks in Sentence \d+\]\n(.*?)(?=\n\[Sentence|$)", re.DOTALL
)
CHUNK_ITEM_PATTERN = re.compile(r"\[(.*?)\]")


def parse_chunks(response: str) -> List[str]:
    if not response:
        return []
    chunk_list: List[str] = []
    for block in CHUNK_BLOCK_PATTERN.findall(response):
        chunk_list.extend(CHUNK_ITEM_PATTERN.findall(block))
    return chunk_list


def extract_chunks_from_text(llm, paragraph_text: str, prompt: str = prompt_chunk_extraction) -> List[str]:
    user_prompt = f"[The Start of Paragraph text]\n{paragraph_text}\n[The End of Paragraph text]"
    response, _tokens = llm.generate(prompt, user_prompt, n=1)
    return parse_chunks(response)


def extract_chunks_from_evidences(llm, evidences: List[list], prompt: str = prompt_chunk_extraction) -> List[List[str]]:
    chunk_lists = []
    for evidence in evidences:
        paragraph_text = "\n".join(evidence[-1])
        chunk_lists.append(extract_chunks_from_text(llm, paragraph_text, prompt))
    return chunk_lists

def run_flat_dataset(
    llm,
    documents: Dict[str, str],
    checkpoint_path: str,
    max_workers: int,
) -> Dict[str, List[str]]:
    results = load_checkpoint(checkpoint_path)
    items = [(key, text) for key, text in documents.items() if str(key) not in results]

    def worker(text: str) -> List[str]:
        return extract_chunks_from_text(llm, text)

    return process_items_concurrently(items, worker, results, checkpoint_path, max_workers=max_workers)


def load_musique_documents(path: str = "datasets/musique/musique.json") -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw_documents = json.load(f)

    documents = {}
    for samp_idx, sample in enumerate(raw_documents):
        paragraphs = [p["paragraph_text"] for p in sample["paragraphs"]]
        documents[str(samp_idx)] = "\n".join(paragraphs)
    return documents


def run_musique(llm, checkpoint_path: str, max_workers: int) -> dict:
    documents = load_musique_documents()
    return run_flat_dataset(llm, documents, checkpoint_path, max_workers)


def run_arena(llm, document_path: str, checkpoint_path: str, max_workers: int) -> dict:
    with open(document_path, "r", encoding="utf-8") as f:
        raw_documents = json.load(f)

    documents = {key: "\n".join(lines) for key, lines in raw_documents.items()}

    start = time.time()
    results = run_flat_dataset(llm, documents, checkpoint_path, max_workers)
    elapsed_minutes = (time.time() - start) / 60
    logger.info("Time process: %.2f min", elapsed_minutes)
    return results




def run_context_based_dataset(llm, dataset_path: str, checkpoint_path: str, max_workers: int) -> dict:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_documents = json.load(f)

    results = load_checkpoint(checkpoint_path)
    items = [
        (str(samp_idx), doc["context"])
        for samp_idx, doc in enumerate(raw_documents)
        if str(samp_idx) not in results
    ]

    def worker(context: List[list]) -> List[List[str]]:
        return extract_chunks_from_evidences(llm, context)

    return process_items_concurrently(items, worker, results, checkpoint_path, max_workers=max_workers)


def run_hotpotqa(llm, checkpoint_path: str, max_workers: int) -> dict:
    return run_context_based_dataset(llm, "datasets/hotpotqa/hotpotqa.json", checkpoint_path, max_workers)


def run_2wikimultihopqa(llm, checkpoint_path: str, max_workers: int) -> dict:
    return run_context_based_dataset(llm, "datasets/2wiki/2wikimultihopqa.json", checkpoint_path, max_workers)


DATASET_RUNNERS = {
    "musique": lambda llm, config: run_musique(llm, config["destiny_path"], config.get("max_workers", 8)),
    "hotpotqa": lambda llm, config: run_hotpotqa(llm, config["destiny_path"], config.get("max_workers", 8)),
    "2wiki": lambda llm, config: run_2wikimultihopqa(llm, config["destiny_path"], config.get("max_workers", 8)),
    "arena": lambda llm, config: run_arena(
        llm, config["document_path"], config["destiny_path"], config.get("max_workers", 8)
    ),
}


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Extracting chunks for dataset '%s'...", config["dataset"])

    llm = build_llm(config)

    if config["dataset"] == "iclr":
        from src.utils import iclr 

        processed_dataset = open_jsonl(config["document_path"])
        outputs = iclr(llm, processed_dataset)
    elif config["dataset"] in DATASET_RUNNERS:
        outputs = DATASET_RUNNERS[config["dataset"]](llm, config)
    else:
        raise ValueError(f"Dataset not suported: {config['dataset']}")

    with open(config["destiny_path"], "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

    logger.info("Extraction of chunks done!")


if __name__ == "__main__":
    main()