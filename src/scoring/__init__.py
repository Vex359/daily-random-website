"""Content scoring and ranking.

Scores collected content by interestingness to determine
which posts to feature on the daily website.
"""

from src.scoring.interestingness import InterestingnessScorer

__all__ = ["InterestingnessScorer"]
