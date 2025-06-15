from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import ClassificationRequest, ClassificationResponse, ErrorResponse
from app.models.tester_schemas import TestRequest, TestResponse, TestSummary
from app.services.classifier import LangChainHierarchicalClassifier
from app.services.tester import CSVClassifierTester
from app.config import get_settings
import os
from datetime import datetime
from app.core.tree_extractor import extract_classification_tree_from_csv

router = APIRouter()
settings = get_settings()

# Load classification tree
categories_csv = "Categories Combinations(Sheet1).csv"
classification_tree = extract_classification_tree_from_csv(categories_csv)

# Initialize classifier with the tree
classifier = LangChainHierarchicalClassifier(classification_tree=classification_tree)

@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def classify_text(request: ClassificationRequest):
    """
    Classify the input text using the hierarchical classifier.
    """
    try:
        if not classifier.classification_tree:
            raise HTTPException(
                status_code=500,
                detail="Classification tree not initialized. Please set the classification tree first."
            )

        result = classifier.classify(request.text)
        return ClassificationResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )

@router.post(
    "/test",
    response_model=TestResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def run_classification_test(request: TestRequest, background_tasks: BackgroundTasks):
    """
    Run classification tests on a CSV file and return the results.
    """
    try:
        if not os.path.exists(request.csv_file_path):
            raise HTTPException(
                status_code=400,
                detail=f"CSV file not found: {request.csv_file_path}"
            )

        if not classifier.classification_tree:
            raise HTTPException(
                status_code=500,
                detail="Classification tree not initialized. Please set the classification tree first."
            )

        # Create output directory if it doesn't exist
        output_dir = "test_results"
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filenames using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{timestamp}.csv")
        report_file = os.path.join(output_dir, f"report_{timestamp}.txt")

        # Initialize tester
        tester = CSVClassifierTester(classifier, request.csv_file_path)

        # Run tests
        await tester.run_all_tests(max_entries=request.max_entries)

        # Calculate metrics
        tester.calculate_accuracy_metrics()

        # Save results in background
        background_tasks.add_task(tester.save_results, results_file)
        background_tasks.add_task(tester.save_accuracy_report, report_file)

        # Get summary
        summary = tester.get_summary()

        return TestResponse(
            success=True,
            message="Test completed successfully",
            summary=summary
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        ) 