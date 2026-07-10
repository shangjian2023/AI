"""Tests for disjoint trigger-search and final-validation prompts."""
from src.detection.scorer import BASE_QUESTIONS, VALIDATION_QUESTIONS


def test_validation_questions_are_nonempty_and_disjoint_from_search_questions():
    assert VALIDATION_QUESTIONS
    assert set(BASE_QUESTIONS).isdisjoint(VALIDATION_QUESTIONS)
