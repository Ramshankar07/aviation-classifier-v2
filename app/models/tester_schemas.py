from pydantic import BaseModel
from typing import Dict, Optional, List

class TestRequest(BaseModel):
    csv_file_path: str
    max_entries: Optional[int] = None

class TestResponse(BaseModel):
    success: bool
    message: str
    summary: Optional[Dict] = None

class AccuracyMetrics(BaseModel):
    overall_accuracy: Dict[str, float]
    summary_stats: Dict[str, float]
    total_entries: int

class TestSummary(BaseModel):
    accuracy_metrics: AccuracyMetrics
    results_file: str
    report_file: str

class TestResult(BaseModel):
    log_notes: str
    expected: Dict[str, str]
    predicted: Dict[str, str]
    is_correct: bool

class TestMetrics(BaseModel):
    total_tests: int
    correct_classifications: int
    accuracy: float
    level_accuracy: Dict[str, float] 