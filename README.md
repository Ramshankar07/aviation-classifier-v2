# Aviation Classifier API

A FastAPI-based API for classifying aviation-related text using hierarchical classification with LangChain and Together AI.

## Features

- Hierarchical classification of aviation-related text
- Department, Category, and Subcategory classification
- Operational details classification
- FastAPI-based REST API
- LangChain integration with Together AI
- ChromaDB vector store for semantic search
- CSV-based testing and evaluation
- Detailed accuracy reports and metrics

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd aviation-classifier-v2
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your configuration:
```env
TOGETHER_API_KEY=your_together_api_key
VECTORSTORE_PATH=./vectorstore
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-Turbo
TEMPERATURE=0.1
MAX_TOKENS=500
```

## Running the API

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Usage

### Classify Text

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "Gate 15 jetbridge malfunction causing boarding delays"}'
```

Response:
```json
{
    "Department": "Infrastructure",
    "Category": "Building Operations and Maintenance",
    "Sub_Category": "Bridges/Gates",
    "Operational_Entity": "Jetbridge",
    "Status": "Malfunction",
    "Operational_Trigger": "Equipment Failure",
    "Location_Type": "Gate",
    "Location": "Gate 15"
}
```

### Run Classification Tests

```bash
curl -X POST "http://localhost:8000/api/v1/test" \
     -H "Content-Type: application/json" \
     -d '{
         "csv_file_path": "path/to/your/test_data.csv",
         "max_entries": 100
     }'
```

Response:
```json
{
    "success": true,
    "message": "Test completed successfully",
    "summary": {
        "overall_accuracy": {
            "Department": 85.5,
            "Category": 82.3,
            "Sub_Category": 78.9,
            "Operational_Entity": 80.1,
            "Status": 83.4,
            "Operational_Trigger": 79.8,
            "Location_Type": 81.2,
            "Location": 84.5,
            "Overall": 75.6
        },
        "summary_stats": {
            "mean_accuracy": 81.4,
            "std_accuracy": 2.8,
            "min_accuracy": 75.6,
            "max_accuracy": 85.5
        },
        "total_entries": 100
    }
}
```

The test results will be saved in the `test_results` directory:
- `results_[timestamp].csv`: Detailed classification results
- `report_[timestamp].txt`: Detailed accuracy report

## CSV Test File Format

The test CSV file should have the following columns:
- `Log Notes`: The text to classify
- `Department`: Actual department
- `Category`: Actual category
- `Sub Category`: Actual subcategory
- `Operational Element (Issue/Condition/Cause/Activity)`: Actual operational entity
- `Status`: Actual status
- `Operational Trigger`: Actual operational trigger
- `Location Type`: Actual location type
- `Location`: Actual location

## Project Structure

```
aviation-classifier-v2/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── llm.py
│   ├── models/
│   │   ├── schemas.py
│   │   └── tester_schemas.py
│   ├── services/
│   │   ├── classifier.py
│   │   └── tester.py
│   └── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Dependencies

- FastAPI: Web framework
- LangChain: Framework for LLM applications
- Together AI: LLM provider
- ChromaDB: Vector store
- Sentence Transformers: Text embeddings
- Pydantic: Data validation
- Uvicorn: ASGI server
- Pandas: Data manipulation
- NumPy: Numerical computations

## License

[Your License] 