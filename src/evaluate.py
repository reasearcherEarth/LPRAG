import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from tqdm import tqdm
import string
import re
import logging
logger = logging.getLogger(__name__)


def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class Evaluator:
    def __init__(self,predictions_path):
        self.predictions_path = predictions_path
        self.prediction_results = self.load_predictions()

    def load_predictions(self):
        with open(self.predictions_path, encoding="utf-8") as f:
            prediction_results = json.load(f)
        # prediction_results = json.load(open(self.predictions_path), encoding="utf-8")
        return prediction_results

    def calculate_contain(self,pre_answers,gold_ans):
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers)
        s2 = normalize_answer(gold_ans)
        if s2 in s1 or s1 in s2:
            return 1
        else:
            return 0
        
    def calculate_f1(self,pre_answers,gold_ans) -> float:
        """F1 baseado em sobreposição de tokens."""
        
        if pre_answers is None or pre_answers == "" or (isinstance(pre_answers, str) and pre_answers.strip() == ""):
            return 0            
        if gold_ans is None or gold_ans == "" or (isinstance(gold_ans, str) and gold_ans.strip() == ""):
            return 0
        s1 = normalize_answer(pre_answers).split()
        s2 = normalize_answer(gold_ans).split()
        
        common = Counter(s1) & Counter(s2)
        num_same = sum(common.values())
        
        if len(s1) == 0 or len(s2) == 0:
            return float(s1 == s2)
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(s1)
        recall = num_same / len(s2)
        
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
            
    def evaluate_sig_sample(self,idx,prediction):
        pre_answer = prediction["pred_answer"]
        gold_ans = prediction["gold_answer"]
        f1 = self.calculate_f1(pre_answer, gold_ans)
        contain_acc = self.calculate_contain(pre_answer, gold_ans)
        return idx, f1, contain_acc

    def evaluate(self,max_workers):
        f1_scores = [0.0] * len(self.prediction_results)
        contain_scores = [0.0] * len(self.prediction_results)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_sig_sample, idx, pred): idx 
                for idx, pred in enumerate(self.prediction_results)
            }

            completed = 0
            total_f1 = 0.0
            total_contain_score = 0.0
            pbar = tqdm(total=len(futures), desc="Evaluating samples", unit="sample")
            for future in as_completed(futures):
                idx, f1, contain_acc  = future.result()
                f1_scores[idx] = f1
                contain_scores[idx] = contain_acc
                self.prediction_results[idx]["f1"] = f1
                self.prediction_results[idx]["contain_accuracy"] = contain_acc
                total_f1 += f1
                total_contain_score += contain_acc
                completed += 1
                current_f1 = total_f1 / completed
                current_contain_acc = total_contain_score / completed
                pbar.set_postfix({
                    'f1': f'{current_f1:.3f}',
                    'Contain_Acc': f'{current_contain_acc:.3f}'
                })
                pbar.update(1)
            pbar.close()

        f1 = sum(f1_scores) / len(f1_scores)
        contain_accuracy = sum(contain_scores) / len(contain_scores)

        logger.info(f"Evaluation Results:")
        logger.info(f"  f1: {f1:.4f} ({sum(f1_scores)}/{len(f1_scores)})")
        logger.info(f"  Contain Accuracy: {contain_accuracy:.4f} ({sum(contain_scores)}/{len(contain_scores)})")
        with open(self.predictions_path, "w", encoding="utf-8") as f:
            json.dump(self.prediction_results, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(os.path.dirname(self.predictions_path), "evaluation_results.json"), "w", encoding="utf-8") as f:
            json.dump({"f1": f1, "contain_accuracy": contain_accuracy}, f, ensure_ascii=False, indent=4)
        return f1, contain_accuracy