# Aviation Classifier API 🛩️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green)](https://fastapi.tiangolo.com/)

A powerful FastAPI-based API for hierarchical classification of aviation-related text using LangChain and Together AI. This system provides detailed classification of aviation incidents, maintenance issues, and operational events through multiple hierarchical levels.

## ✨ Features

- **Hierarchical Classification**: Multi-level classification system for aviation text
  - Department → Category → Subcategory
  - Operational details (Entity, Status, Trigger)
  - Location information (Type and specific Location)
- **Advanced AI Integration**
  - LangChain framework for robust LLM applications
  - Together AI for high-performance language models
  - ChromaDB vector store for efficient semantic search
- **Comprehensive Testing**
  - CSV-based test suite
  - Detailed accuracy metrics and reports
  - Performance benchmarking
- **Developer-Friendly**
  - RESTful API with OpenAPI documentation
  - Easy-to-use endpoints
  - Comprehensive error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
```bash

git clone https://github.com/Ramshankar07/aviation-classifier-v2.git

cd aviation-classifier-v2
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the root directory:
```env
TOGETHER_API_KEY=your_together_api_key
VECTORSTORE_PATH=./vectorstore
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-Turbo
TEMPERATURE=0.1
MAX_TOKENS=500
```

### Setting Up the Classifier

1. The project includes a sample classification tree in `sample_classification_tree.csv`. This file contains:
   - Department (Level 1)
   - Category (Level 2)
   - Subcategory (Level 3)
   - Operational Entities
   - Statuses
   - Triggers
   - Location Types
   - Locations

2. Use the provided `setup_classifier.py` script to initialize the classifier:
```bash
python setup_classifier.py
```

3. The script will:
   - Initialize the LangChainHierarchicalClassifier
   - Load the classification tree from the CSV
   - Set up the classifier with the tree
   - Run a test classification

4. You can modify `sample_classification_tree.csv` to add more:
   - Departments (e.g., Infrastructure, Security, Safety)
   - Categories (e.g., Building Operations, Security Screening)
   - Subcategories (e.g., Gate Assignment, Security Checks)
   - Operational details and locations

Example of adding a new classification:
```csv
Index,Level_1,Level_2,Level_3,Operational_Entities,Statuses,Triggers,Location_Types,Locations
1,Infrastructure,Building Operations,Gate Assignment,Jetbridge,Malfunction,Equipment Failure,Gate,Gate 15
```

## 🏃‍♂️ Running the API

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📚 API Usage

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

## 📊 Test Results Format

The test results will be saved in the `test_results` directory:
- `results_[timestamp].csv`: Detailed classification results
- `report_[timestamp].txt`: Detailed accuracy report

### CSV Test File Format

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

## 🏗️ Project Structure

```
aviation-classifier-v2/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   └── llm.py            # LLM configuration
│   ├── models/
│   │   ├── schemas.py        # Pydantic models
│   │   └── tester_schemas.py # Test schemas
│   ├── services/
│   │   ├── classifier.py     # Classification logic
│   │   └── tester.py         # Testing utilities
│   └── config.py             # App configuration
├── main.py                   # Application entry point
├── requirements.txt          # Project dependencies
├── sample_classification_tree.csv  # Sample classification data
├── setup_classifier.py       # Classifier setup script
└── README.md                 # Project documentation
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your Together AI API key is correctly set in the `.env` file
   - Check if the API key has sufficient permissions

2. **Vector Store Errors**
   - Verify the `VECTORSTORE_PATH` exists and is writable
   - Clear the vector store directory if corrupted

3. **Model Loading Issues**
   - Confirm the model name is correct in `.env`
   - Check internet connection for model download

4. **Classification Tree Issues**
   - Ensure the CSV file follows the correct format
   - Check for any missing or malformed data in the CSV
   - Verify the classification tree is properly loaded

## 📦 Dependencies

- FastAPI: Web framework
- LangChain: Framework for LLM applications
- Together AI: LLM provider
- ChromaDB: Vector store
- Sentence Transformers: Text embeddings
- Pydantic: Data validation
- Uvicorn: ASGI server
- Pandas: Data manipulation
- NumPy: Numerical computations
