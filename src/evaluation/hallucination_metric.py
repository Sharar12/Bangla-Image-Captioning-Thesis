from typing import List, Set

class HallucinationDetector:
    def evaluate(self, generated_caption: str, ground_truth_objects: Set[str]) -> float:
        words = set(generated_caption.split())
        hallucinated = [w for w in words if w in {"গাড়ি", "মানুষ", "বিড়াল", "কুকুর"} and w not in ground_truth_objects]
        return len(hallucinated) / max(1, len(words))
