# src/agents/complexity_checker.py
from typing import Dict, List
from .base_agent import BaseAgent


class ComplexityChecker(BaseAgent):
    """Agent responsible for classifying medical query complexity"""

    INSTRUCTION = """You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.

Given a medical query, you need to classify it into one of three complexity levels:

1) Low: A PCP or general physician can answer this simple medical knowledge checking question without relying heavily on consulting other specialists.
2) Moderate: A PCP or general physician can answer this question in consultation with other specialists in a team.
3) High: Team of multi-departmental specialists can answer the question which requires specialists consulting to another department (requires a lot of team effort to treat the case).

Consider factors such as:
- Number of medical specialties involved
- Complexity of differential diagnosis
- Need for multi-modal interpretation (images, labs, etc.)
- Severity and urgency of the condition
- Required coordination between departments

Respond with ONLY one of these words: low, moderate, or high"""

    def __init__(self, logger=None):
        # Few-shot examples for complexity classification
        few_shot_examples = [
            {
                "question": """Question: A 55-year-old female patient with type 2 diabetes visits her PCP for a routine check-up. Her fasting glucose is 145 mg/dL. She takes metformin 500mg twice daily. What adjustment should be made?
Options: A) Increase metformin to 1000mg twice daily B) Add insulin C) No change needed D) Add sulfonylurea""",
                "answer": "low"
            },
            {
                "question": """Question: A 40-year-old male presents to ED with fever, severe headache, and muscle pain after recent travel to Southeast Asia. He has neck stiffness and a petechial rash. Blood tests show thrombocytopenia.
Options: A) Dengue fever B) Meningococcal meningitis C) Malaria D) Typhoid fever""",
                "answer": "moderate"
            },
            {
                "question": """Question: A 63-year-old woman with gradual onset double vision, droopy right eyelid, right pupil 6mm poorly reactive to light. Also has jaw claudication and temporal headache. ESR is elevated.
Options: A) Myasthenia gravis B) Temporal arteritis with CN III palsy C) Diabetic neuropathy D) Brain tumor""",
                "answer": "high"
            }
        ]

        super().__init__(
            role="Complexity Checker",
            instruction=self.INSTRUCTION,
            few_shot_examples=few_shot_examples,
            logger=logger
        )

    def check_complexity(self, question: str) -> str:
        """Classify the complexity of a medical question"""
        prompt = f"Medical Query: {question}\n\nComplexity Level:"

        response, token_info = self.chat(
            prompt, temperature=0.3)  # Lower temp for consistency

        # Extract complexity level
        complexity = response.strip().lower()

        # Validate response
        if complexity not in ["low", "moderate", "high"]:
            print(
                f"Warning: Invalid complexity '{complexity}', defaulting to 'moderate'")
            complexity = "moderate"

        # Store for logging purposes
        self.current_complexity = complexity

        return complexity
