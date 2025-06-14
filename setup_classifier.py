from app.services.classifier import LangChainHierarchicalClassifier
from app.core.tree_extractor import extract_classification_tree_from_csv

def setup_classifier():
    # Initialize the classifier
    classifier = LangChainHierarchicalClassifier()
    
    # Extract classification tree from CSV
    classification_tree = extract_classification_tree_from_csv('sample_classification_tree.csv')
    
    # Set the classification tree in the classifier
    classifier.set_classification_tree(classification_tree)
    
    return classifier

if __name__ == "__main__":
    # Example usage
    classifier = setup_classifier()
    
    # Test classification
    test_text = "Gate 15 jetbridge malfunction causing boarding delays"
    result = classifier.classify(test_text)
    print("Classification Result:", result) 