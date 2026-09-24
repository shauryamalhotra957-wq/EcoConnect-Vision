"""
Classification Debounce & Temporal Consistency Filter.
Smooths high-speed video frame classifications across conveyor belt scans
to prevent actuator flapping and false diversion triggers.
"""
from typing import Dict, Optional, Tuple

class ClassificationDebouncer:
    def __init__(self, persistence_threshold: int = 3, min_confidence: float = 0.65):
        self.persistence_threshold = persistence_threshold
        self.min_confidence = min_confidence
        self.candidate_label: Optional[str] = None
        self.candidate_count: int = 0
        self.committed_label: Optional[str] = None

    def update(self, detected_label: str, confidence: float) -> Tuple[Optional[str], bool]:
        """
        Processes a single frame inference.
        Returns: (current_committed_label, is_newly_triggered)
        """
        if confidence < self.min_confidence:
            return self.committed_label, False

        if detected_label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = detected_label
            self.candidate_count = 1

        if self.candidate_count >= self.persistence_threshold:
            if self.committed_label != self.candidate_label:
                self.committed_label = self.candidate_label
                return self.committed_label, True

        return self.committed_label, False

    def reset(self):
        self.candidate_label = None
        self.candidate_count = 0
        self.committed_label = None
