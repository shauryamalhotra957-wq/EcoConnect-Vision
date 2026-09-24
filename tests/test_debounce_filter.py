import pytest
from debounce_filter import ClassificationDebouncer

def test_debouncer_ignores_low_confidence():
    debouncer = ClassificationDebouncer(persistence_threshold=3, min_confidence=0.7)
    label, triggered = debouncer.update("cardboard", 0.5)
    assert label is None
    assert triggered is False

def test_debouncer_triggers_after_threshold():
    debouncer = ClassificationDebouncer(persistence_threshold=3, min_confidence=0.7)
    
    assert debouncer.update("plastic", 0.8) == (None, False)
    assert debouncer.update("plastic", 0.85) == (None, False)
    label, triggered = debouncer.update("plastic", 0.9)
    assert label == "plastic"
    assert triggered is True
    
    # Continued identical detections do not re-trigger
    label2, triggered2 = debouncer.update("plastic", 0.88)
    assert label2 == "plastic"
    assert triggered2 is False

def test_debouncer_flapping_rejected():
    debouncer = ClassificationDebouncer(persistence_threshold=3, min_confidence=0.7)
    debouncer.update("metal", 0.8)
    debouncer.update("glass", 0.8)
    debouncer.update("metal", 0.8)
    assert debouncer.committed_label is None
