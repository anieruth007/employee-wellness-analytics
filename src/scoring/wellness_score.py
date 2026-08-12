"""Combined Wellness Score (0-100) from Stress Index and Cognitive Load Index.

    wellness_score = 100 - ((stress_index + (100 - cognitive_load_index)) / 2)

Verified against the four corner cases from the reframe spec:
  High stress (100) + Low cognitive load (0)   -> 0   (low wellness)
  Low stress (0)    + High cognitive load (100) -> 100 (high wellness)
  Low stress (0)    + Low cognitive load (0)    -> 50  (medium)
  High stress (100) + High cognitive load (100) -> 50  (medium — "burned out risk": pushing
                                                          through despite high stress)
"""


def compute_wellness_score(stress_index: float, cognitive_load_index: float) -> float:
    return 100.0 - ((stress_index + (100.0 - cognitive_load_index)) / 2.0)
