import asyncio
import os
from datetime import datetime
from app.services.tester import CSVClassifierTester
from app.services.classifier import LangChainHierarchicalClassifier
from app.core.tree_extractor import extract_classification_tree_from_csv

async def main():
    # Define paths
    categories_csv = "Categories Combinations(Sheet1).csv"  
    input_csv = "Live Notes Categorization - Completed(YLW Ops Log).csv"  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"data/results/live_notes_results_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Loading classification tree from: {categories_csv}")
    classification_tree = extract_classification_tree_from_csv(categories_csv)
    
    if not classification_tree:
        print("Error: Failed to load classification tree!")
        return
    
    print("Classification tree loaded successfully!")
    
    # Initialize the classifier with the classification tree
    classifier = LangChainHierarchicalClassifier(classification_tree=classification_tree)
    
    # Initialize the tester
    tester = CSVClassifierTester(classifier)
    
    print(f"Starting live notes processing...")
    print(f"Input file: {input_csv}")
    print(f"Output file: {output_csv}")
    
    # Process the live notes
    success = await tester.process_live_notes(
        input_csv_path=input_csv,
        output_csv_path=output_csv
    )
    
    if success:
        print("Processing completed successfully!")
        print(f"Results saved to: {output_csv}")
    else:
        print("Processing failed!")

if __name__ == "__main__":
    asyncio.run(main()) 