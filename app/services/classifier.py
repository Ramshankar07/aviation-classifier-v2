import json
from typing import Dict, List, Any
from pathlib import Path

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from app.core.llm import TogetherLLM
from app.config import get_settings

settings = get_settings()

class LangChainHierarchicalClassifier:
    def __init__(self, classification_tree: Dict[str, Any] = None):
        """
        Initialize the hierarchical classifier
        
        Args:
            classification_tree: Optional dictionary containing the hierarchical classification structure
        """
        self.classification_tree = classification_tree or {}
        # Initialize LLM
        self.llm = TogetherLLM()

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load ChromaDB vectorstore
        self.vectorstore = Chroma(
            persist_directory=str(Path(settings.VECTORSTORE_PATH)),
            embedding_function=self.embeddings
        )

        # Initialize RetrievalQA chains for each level
        self._setup_qa_chains()

    def _setup_qa_chains(self):
        """Setup RetrievalQA chains for each classification level"""
        # Department Classification Chain
        department_prompt = PromptTemplate(
            template="""You are a Department Classification Agent.
Use the following context and your knowledge to classify the input into the appropriate department.

Context from knowledge base:
{context}

Department descriptions:
- Operations: General terminal or airside activity coordination, gate management, aircraft parking, delays, ramp activities, and passenger or airline service support.
- Infrastructure: Issues or activities related to terminal buildings, hangars, power systems, elevators, HVAC, plumbing, electrical systems, or other fixed airport facilities.
- Security: Personnel checks, surveillance, breaches, unauthorized access, suspicious persons or packages, ID checks, or law enforcement involvement.
- Safety: Hazards, near-misses, safety observations, foreign object debris (FOD), safety violations, slips, trips, and falls.
- Emergency: Medical responses, aircraft incidents, fuel spills, fire alarms, evacuations, emergency drills or emergency coordination.
- Miscellaneous: Items that clearly do not belong to any other department or are too vague or unrelated to classify.
- Field Maintenance: Runway/taxiway repairs, lighting issues, snow/ice removal, rubber removal, paint marking, grass cutting, or surface inspections.
- Firehall: Activities specifically related to airport fire crew (ARFF), fire vehicle maintenance, readiness checks, or fire station operations.
- Airport Security: TSA or contracted security screening operations, baggage screening, access control point issues, security queue management.

Input to classify: {question}

Return ONLY the department name that best fits the context. Choose from the departments list above.
Department:""",
            input_variables=["context", "question"]
        )

        self.department_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": department_prompt}
        )

        # Category Classification Chain
        category_prompt = PromptTemplate(
            template="""You are a Category Classification Agent.
Use the following context to help classify the input into the appropriate category within the given department.

Context from knowledge base:
{context}

Input to classify: {question}

Return ONLY the category name that best fits within the department. Choose from the available categories list.
Category:""",
            input_variables=["context", "question"]
        )

        self.category_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": category_prompt}
        )

        # Subcategory Classification Chain
        subcategory_prompt = PromptTemplate(
            template="""You are a Sub-Category Classification Agent.
Use the following context to help classify the input into the appropriate sub-category.

Context from knowledge base:
{context}

Input to classify: {question}

Return ONLY the sub-category name that best fits. Choose from the available sub-categories list.
Sub-Category:""",
            input_variables=["context", "question"]
        )

        self.subcategory_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": subcategory_prompt}
        )

        # Final Details Classification Chain
        final_prompt = PromptTemplate(
            template="""You are a Final Classification Agent.
Use the following context to help classify the input into the remaining operational fields.

Context from knowledge base:
{context}

Input to classify: {question}

Analyze the input and return a JSON object with these exact keys:
- "Operational Entity": [choose from operational entities list]
- "Status": [choose from statuses list]
- "Operational Trigger": [choose from triggers list]
- "Location Type": [choose from location types list]
- "Location": [choose from locations list]

Return only the JSON object, no explanations.
JSON:""",
            input_variables=["context", "question"]
        )

        self.final_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": final_prompt}
        )

    def get_categories_for_department(self, department: str) -> List[str]:
        if not self.classification_tree:
            return []
        return list(self.classification_tree.get(department, {}).keys())

    def get_subcategories_for_category(self, department: str, category: str) -> List[str]:
        if not self.classification_tree:
            return []
        return list(self.classification_tree.get(department, {}).get(category, {}).keys())

    def get_operational_options(self, department: str, category: str, subcategory: str) -> Dict[str, List[str]]:
        if not self.classification_tree:
            return {
                'operational_entities': [],
                'statuses': [],
                'triggers': [],
                'location_types': [],
                'locations': []
            }

        path = self.classification_tree.get(department, {}).get(category, {}).get(subcategory, {})
        return {
            'operational_entities': path.get('operational_entities', []),
            'statuses': path.get('statuses', []),
            'triggers': path.get('triggers', []),
            'location_types': path.get('location_types', []),
            'locations': path.get('locations', [])
        }

    def classify_department(self, input_text: str) -> str:
        """Level 1: Classify Department using RetrievalQA"""
        try:
            departments = list(self.classification_tree.keys()) if self.classification_tree else []
            query_with_context = f"Input to classify: {input_text}\nAvailable departments: {departments}"
            result = self.department_qa.invoke({"query": query_with_context})
            answer = result.get("result", "")

            if self.classification_tree:
                for dept in departments:
                    if dept.lower() in answer.lower() or answer.lower() in dept.lower():
                        return dept
                return "NA"
            return answer.strip()

        except Exception as e:
            print(f"Error in department classification: {e}")
            if self.classification_tree:
                departments = list(self.classification_tree.keys())
                return departments[0] if departments else "NA"
            return "NA"

    def classify_category(self, input_text: str, department: str) -> str:
        """Level 2: Classify Category using RetrievalQA"""
        available_categories = self.get_categories_for_department(department)
        if not available_categories:
            return "NA"

        try:
            query_with_context = f"Input to classify: {input_text}\nDepartment: {department}\nAvailable categories: {available_categories}"
            result = self.category_qa.invoke({"query": query_with_context})
            answer = result.get("result", "")

            for cat in available_categories:
                if cat.lower() in answer.lower() or answer.lower() in cat.lower():
                    return cat
            return "NA"

        except Exception as e:
            print(f"Error in category classification: {e}")
            return "NA"

    def classify_subcategory(self, input_text: str, department: str, category: str) -> str:
        """Level 3: Classify Subcategory using RetrievalQA"""
        available_subcategories = self.get_subcategories_for_category(department, category)
        if not available_subcategories:
            return "NA"

        try:
            query_with_context = f"Input to classify: {input_text}\nDepartment: {department}\nCategory: {category}\nAvailable subcategories: {available_subcategories}"
            result = self.subcategory_qa.invoke({"query": query_with_context})
            answer = result.get("result", "")

            for subcat in available_subcategories:
                if subcat.lower() in answer.lower() or answer.lower() in subcat.lower():
                    return subcat
            return "NA"

        except Exception as e:
            print(f"Error in subcategory classification: {e}")
            return "NA"

    def classify_final_details(self, input_text: str, department: str, category: str, subcategory: str) -> Dict[str, str]:
        """Level 4: Final Classification using RetrievalQA"""
        operational_options = self.get_operational_options(department, category, subcategory)

        try:
            query_with_context = f"Input to classify: {input_text}\nDepartment: {department}\nCategory: {category}\nSubcategory: {subcategory}\nAvailable Options:\nOperational Entities: {operational_options['operational_entities']}\nStatuses: {operational_options['statuses']}\nTriggers: {operational_options['triggers']}\nLocation Types: {operational_options['location_types']}\nLocations: {operational_options['locations']}"

            result = self.final_qa.invoke({"query": query_with_context})
            answer = result.get("result", "")

            try:
                final_classification = json.loads(answer)
                validated_result = {}

                # Validate each field
                for field, options in [
                    ("Operational Entity", operational_options['operational_entities']),
                    ("Status", operational_options['statuses']),
                    ("Operational Trigger", operational_options['triggers']),
                    ("Location Type", operational_options['location_types']),
                    ("Location", operational_options['locations'])
                ]:
                    value = final_classification.get(field, "")
                    for option in options:
                        if option.lower() in value.lower() or value.lower() in option.lower():
                            validated_result[field] = option
                            break
                    else:
                        validated_result[field] = options[0] if options else "NA"

                return validated_result

            except json.JSONDecodeError:
                return {
                    "Operational Entity": operational_options['operational_entities'][0] if operational_options['operational_entities'] else "NA",
                    "Status": operational_options['statuses'][0] if operational_options['statuses'] else "NA",
                    "Operational Trigger": operational_options['triggers'][0] if operational_options['triggers'] else "NA",
                    "Location Type": operational_options['location_types'][0] if operational_options['location_types'] else "NA",
                    "Location": operational_options['locations'][0] if operational_options['locations'] else "NA"
                }

        except Exception as e:
            print(f"Error in final classification: {e}")
            return {
                "Operational Entity": operational_options['operational_entities'][0] if operational_options['operational_entities'] else "NA",
                "Status": operational_options['statuses'][0] if operational_options['statuses'] else "NA",
                "Operational Trigger": operational_options['triggers'][0] if operational_options['triggers'] else "NA",
                "Location Type": operational_options['location_types'][0] if operational_options['location_types'] else "NA",
                "Location": operational_options['locations'][0] if operational_options['locations'] else "NA"
            }

    async def classify(self, text: str) -> Dict[str, str]:
        """
        Classify the input text using the hierarchical structure
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing classification results for each level
        """
        department = self.classify_department(text)
        category = self.classify_category(text, department)
        subcategory = self.classify_subcategory(text, department, category)
        final_details = self.classify_final_details(text, department, category, subcategory)

        return {
            "Department": department,
            "Category": category,
            "Sub_Category": subcategory,
            **final_details
        } 