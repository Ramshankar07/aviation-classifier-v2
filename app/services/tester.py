import pandas as pd
import asyncio
import json
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np
from datetime import datetime
from app.services.classifier import LangChainHierarchicalClassifier

class CSVClassifierTester:
    def __init__(self, classifier: LangChainHierarchicalClassifier, csv_file_path: str):
        """
        Initialize the CSV tester with classifier and CSV file path
        """
        self.classifier = classifier
        self.csv_file_path = csv_file_path
        self.df = None
        self.results = []
        self.accuracy_metrics = {}

    def load_csv(self):
        """Load the CSV file"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Successfully loaded CSV with {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            print("No data loaded")
            return False

        # Remove rows with empty log notes
        self.df = self.df.dropna(subset=['Log Notes'])
        self.df = self.df[self.df['Log Notes'].str.strip() != '']

        # Clean column names (remove extra spaces)
        self.df.columns = self.df.columns.str.strip()

        print(f"After preprocessing: {len(self.df)} rows with valid log notes")
        return True

    async def test_single_entry(self, log_note: str, actual_values: Dict[str, str], index: int):
        """Test classification for a single log note entry"""
        try:
            print(f"Processing entry {index + 1}/{len(self.df)}: {log_note[:50]}...")

            # Get prediction from classifier
            predicted = self.classifier.classify(log_note)

            # Create result entry
            result_entry = {
                'Index': index,
                'Log_Notes': log_note,
                'Actual_Department': actual_values.get('Department', 'N/A'),
                'Predicted_Department': predicted.get('Department', 'N/A'),
                'Actual_Category': actual_values.get('Category', 'N/A'),
                'Predicted_Category': predicted.get('Category', 'N/A'),
                'Actual_Sub_Category': actual_values.get('Sub Category', 'N/A'),
                'Predicted_Sub_Category': predicted.get('Sub Category', 'N/A'),
                'Actual_Operational_Entity': actual_values.get('Operational Element (Issue/Condition/Cause/Activity)', 'N/A'),
                'Predicted_Operational_Entity': predicted.get('Operational Entity', 'N/A'),
                'Actual_Status': actual_values.get('Status', 'N/A'),
                'Predicted_Status': predicted.get('Status', 'N/A'),
                'Actual_Operational_Trigger': actual_values.get('Operational Trigger', 'N/A'),
                'Predicted_Operational_Trigger': predicted.get('Operational Trigger', 'N/A'),
                'Actual_Location_Type': actual_values.get('Location Type', 'N/A'),
                'Predicted_Location_Type': predicted.get('Location Type', 'N/A'),
                'Actual_Location': actual_values.get('Location', 'N/A'),
                'Predicted_Location': predicted.get('Location', 'N/A'),
            }

            # Calculate individual accuracies for this entry
            result_entry.update({
                'Department_Match': 1 if self._normalize_text(result_entry['Actual_Department']) == self._normalize_text(result_entry['Predicted_Department']) else 0,
                'Category_Match': 1 if self._normalize_text(result_entry['Actual_Category']) == self._normalize_text(result_entry['Predicted_Category']) else 0,
                'Sub_Category_Match': 1 if self._normalize_text(result_entry['Actual_Sub_Category']) == self._normalize_text(result_entry['Predicted_Sub_Category']) else 0,
                'Operational_Entity_Match': 1 if self._normalize_text(result_entry['Actual_Operational_Entity']) == self._normalize_text(result_entry['Predicted_Operational_Entity']) else 0,
                'Status_Match': 1 if self._normalize_text(result_entry['Actual_Status']) == self._normalize_text(result_entry['Predicted_Status']) else 0,
                'Operational_Trigger_Match': 1 if self._normalize_text(result_entry['Actual_Operational_Trigger']) == self._normalize_text(result_entry['Predicted_Operational_Trigger']) else 0,
                'Location_Type_Match': 1 if self._normalize_text(result_entry['Actual_Location_Type']) == self._normalize_text(result_entry['Predicted_Location_Type']) else 0,
                'Location_Match': 1 if self._normalize_text(result_entry['Actual_Location']) == self._normalize_text(result_entry['Predicted_Location']) else 0,
            })

            # Overall accuracy (all fields must match)
            result_entry['Overall_Match'] = 1 if all([
                result_entry['Department_Match'],
                result_entry['Category_Match'],
                result_entry['Sub_Category_Match'],
                result_entry['Operational_Entity_Match'],
                result_entry['Status_Match'],
                result_entry['Operational_Trigger_Match'],
                result_entry['Location_Type_Match'],
                result_entry['Location_Match']
            ]) else 0

            return result_entry

        except Exception as e:
            print(f"Error processing entry {index}: {e}")
            return None

    def _normalize_text(self, text):
        """Normalize text for comparison"""
        if pd.isna(text) or text is None:
            return 'n/a'
        return str(text).lower().strip()

    async def run_all_tests(self, max_entries=None):
        """Run classification tests on all entries"""
        if not self.load_csv() or not self.preprocess_data():
            return False

        # Limit entries if specified
        test_df = self.df.head(max_entries) if max_entries else self.df

        print(f"Starting classification tests on {len(test_df)} entries...")

        for index, row in test_df.iterrows():
            log_note = row['Log Notes']

            # Prepare actual values
            actual_values = {
                'Department': row.get('Department', ''),
                'Category': row.get('Category', ''),
                'Sub Category': row.get('Sub Category', ''),
                'Operational Element (Issue/Condition/Cause/Activity)': row.get('Operational Element (Issue/Condition/Cause/Activity)', ''),
                'Status': row.get('Status', ''),
                'Operational Trigger': row.get('Operational Trigger', ''),
                'Location Type': row.get('Location Type', ''),
                'Location': row.get('Location', '')
            }

            result = await self.test_single_entry(log_note, actual_values, index)
            if result:
                self.results.append(result)

    def calculate_accuracy_metrics(self):
        """Calculate detailed accuracy metrics"""
        if not self.results:
            print("No results to calculate metrics from")
            return

        results_df = pd.DataFrame(self.results)
        total_entries = len(results_df)

        # Calculate accuracy for each field
        field_accuracies = {
            'Department': results_df['Department_Match'].mean() * 100,
            'Category': results_df['Category_Match'].mean() * 100,
            'Sub_Category': results_df['Sub_Category_Match'].mean() * 100,
            'Operational_Entity': results_df['Operational_Entity_Match'].mean() * 100,
            'Status': results_df['Status_Match'].mean() * 100,
            'Operational_Trigger': results_df['Operational_Trigger_Match'].mean() * 100,
            'Location_Type': results_df['Location_Type_Match'].mean() * 100,
            'Location': results_df['Location_Match'].mean() * 100,
            'Overall': results_df['Overall_Match'].mean() * 100
        }

        # Calculate class-wise accuracy for each field
        class_wise_accuracy = {}

        for field in ['Department', 'Category', 'Sub_Category', 'Operational_Entity', 'Status', 'Operational_Trigger', 'Location_Type', 'Location']:
            actual_col = f'Actual_{field}'
            predicted_col = f'Predicted_{field}'
            match_col = f'{field}_Match'

            if actual_col in results_df.columns:
                class_accuracy = {}
                unique_classes = results_df[actual_col].unique()

                for class_name in unique_classes:
                    if pd.notna(class_name) and class_name != 'N/A':
                        class_subset = results_df[results_df[actual_col] == class_name]
                        if len(class_subset) > 0:
                            accuracy = class_subset[match_col].mean() * 100
                            class_accuracy[class_name] = {
                                'accuracy': accuracy,
                                'count': len(class_subset),
                                'correct': class_subset[match_col].sum()
                            }

                class_wise_accuracy[field] = class_accuracy

        self.accuracy_metrics = {
            'overall_accuracy': field_accuracies,
            'class_wise_accuracy': class_wise_accuracy,
            'total_entries': total_entries,
            'summary_stats': {
                'mean_accuracy': np.mean(list(field_accuracies.values())),
                'std_accuracy': np.std(list(field_accuracies.values())),
                'min_accuracy': min(field_accuracies.values()),
                'max_accuracy': max(field_accuracies.values())
            }
        }

    def save_results(self, output_file_path: str):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return False

        try:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_file_path, index=False)
            print(f"Results saved to {output_file_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def save_accuracy_report(self, report_file_path: str):
        """Save detailed accuracy report"""
        if not self.accuracy_metrics:
            print("No accuracy metrics to save")
            return False

        try:
            # Create detailed report
            report = []

            # Overall accuracy summary
            report.append("=== CLASSIFICATION ACCURACY REPORT ===\n")
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"Total entries tested: {self.accuracy_metrics['total_entries']}\n\n")

            # Field-wise accuracy
            report.append("=== FIELD-WISE ACCURACY ===\n")
            for field, accuracy in self.accuracy_metrics['overall_accuracy'].items():
                report.append(f"{field}: {accuracy:.2f}%\n")

            report.append(f"\n=== SUMMARY STATISTICS ===\n")
            stats = self.accuracy_metrics['summary_stats']
            report.append(f"Mean Accuracy: {stats['mean_accuracy']:.2f}%\n")
            report.append(f"Standard Deviation: {stats['std_accuracy']:.2f}%\n")
            report.append(f"Minimum Accuracy: {stats['min_accuracy']:.2f}%\n")
            report.append(f"Maximum Accuracy: {stats['max_accuracy']:.2f}%\n\n")

            # Class-wise accuracy
            report.append("=== CLASS-WISE ACCURACY ===\n")
            for field, classes in self.accuracy_metrics['class_wise_accuracy'].items():
                report.append(f"\n--- {field} ---\n")
                for class_name, metrics in classes.items():
                    report.append(f"{class_name}: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['count']})\n")

            # Save report
            with open(report_file_path, 'w', encoding='utf-8') as f:
                f.writelines(report)

            print(f"Accuracy report saved to {report_file_path}")
            return True

        except Exception as e:
            print(f"Error saving accuracy report: {e}")
            return False

    def get_summary(self):
        """Get summary of results"""
        if not self.accuracy_metrics:
            return None

        summary = {
            "overall_accuracy": self.accuracy_metrics['overall_accuracy'],
            "summary_stats": self.accuracy_metrics['summary_stats'],
            "total_entries": self.accuracy_metrics['total_entries']
        }

        return summary 