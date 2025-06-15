import asyncio
import json
from pathlib import Path
from app.services.classifier import LangChainHierarchicalClassifier
from app.services.tester import CSVClassifierTester
from app.core.tree_extractor import extract_classification_tree_from_csv

async def main():
    try:
        # Load classification tree
        categories_csv = "Categories Combinations(Sheet1).csv"
        classification_tree = extract_classification_tree_from_csv(categories_csv)
        
        # Initialize classifier with the tree
        classifier = LangChainHierarchicalClassifier(classification_tree=classification_tree)
        
        # Initialize tester
        tester = CSVClassifierTester(classifier)
        
        # Run tests
        test_file = Path("sample_test_data.csv")
        if not test_file.exists():
            raise FileNotFoundError(f"Test data file not found at {test_file}")
            
        results = await tester.run_tests(str(test_file))
        
        # Calculate metrics
        metrics = tester.calculate_metrics(results)
        
        # Save results
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "detailed_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Print summary
        print("\nTest Results Summary:")
        print(f"Total tests: {metrics['total_tests']}")
        print(f"Correct classifications: {metrics['correct_classifications']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        
        print("\nLevel-wise Accuracy:")
        for level, acc in metrics['level_accuracy'].items():
            print(f"{level}: {acc:.2%}")
            
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 