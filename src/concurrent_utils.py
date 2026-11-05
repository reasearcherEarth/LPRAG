
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Tuple

logger = logging.getLogger(__name__)


def load_checkpoint(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def process_items_concurrently(
    items: Iterable[Tuple[Any, Any]],
    worker_fn: Callable[[Any], Any],
    results: Dict[str, Any],
    checkpoint_path: str,
    max_workers: int = 16,
    checkpoint_every: int = 10,
) -> Dict[str, Any]:
    items = list(items)
    if not items:
        logger.info("Without item new to process.")
        return results

    lock = threading.Lock()
    completed = 0
    total = len(items)

    def _run(key: Any, payload: Any) -> Tuple[Any, Any]:
        try:
            return key, worker_fn(payload)
        except Exception as exc:  # noqa: BLE001 -
            logger.error("Fail process item %s: %s", key, exc)
            return key, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, key, payload): key for key, payload in items}

        for future in as_completed(futures):
            key, result = future.result()
            with lock:
                results[str(key)] = result
                completed += 1
                if completed % checkpoint_every == 0:
                    logger.info("Progress: %d/%d itens done", completed, total)
                    save_checkpoint(checkpoint_path, results)

    save_checkpoint(checkpoint_path, results)
    logger.info("Done: %d/%d itens process.", completed, total)
    return results