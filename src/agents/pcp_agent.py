# src/agents/pcp_agent.py
from typing import Dict, List
from .medical_agent import MedicalAgent


class PCPAgent(MedicalAgent):
    """Primary Care Physician agent for low complexity cases"""

    def __init__(self, agent_id: str = "pcp_1", logger=None):
        super().__init__(
            specialty="Primary Care Physician",
            agent_id=agent_id,
            expertise="general medicine, preventive care, common conditions, initial diagnosis",
            relevance_score=1.0,
            logger=logger
        )

        # Add few-shot examples for common cases
        self.few_shot_examples = [
            {
                "question": "A 45-year-old man with diabetes presents with increased thirst and urination. His fasting glucose is 250 mg/dL. What is the most appropriate management?",
                "answer": "Based on the elevated fasting glucose (250 mg/dL) and classic symptoms of hyperglycemia (polyuria and polydipsia), this patient has uncontrolled diabetes. The most appropriate management would be to intensify diabetes treatment, which could include increasing current medication doses or adding insulin therapy. Answer: A) Intensify diabetes management"
            },
            {
                "question": "A 30-year-old woman complains of fatigue and constipation. Labs show TSH of 12 mU/L. What is the diagnosis?",
                "answer": "The elevated TSH (12 mU/L, normal 0.4-4.0) with symptoms of fatigue and constipation clearly indicates hypothyroidism. This is a straightforward diagnosis based on classic symptoms and confirmatory lab work. Answer: B) Hypothyroidism"
            }
        ]

    def solve_case(self, question: str, options: Dict[str, str] = None,
                   use_cot: bool = True) -> Dict:
        """Solve a medical case using Chain-of-Thought if specified"""

        if use_cot:
            return self._solve_with_cot(question, options)
        else:
            return self.analyze_case(question, options)

    def _solve_with_cot(self, question: str, options: Dict[str, str] = None) -> Dict:
        """Solve using Chain-of-Thought reasoning"""

        # Format question with options
        if options:
            formatted_question = f"{question}\n\nOptions:\n"
            for key, value in sorted(options.items()):
                formatted_question += f"{key}) {value}\n"
        else:
            formatted_question = question

        # Create CoT prompt
        cot_prompt = f"""Let's approach this medical case step-by-step.

{formatted_question}

Please think through this systematically:

Step 1: Identify the key clinical information
Step 2: Consider the pathophysiology
Step 3: Think about the most likely diagnosis
Step 4: Evaluate each option
Step 5: Select the best answer

Let's work through each step:"""

        response, token_info = self.chat(cot_prompt, temperature=0.7)

        # Parse the response
        analysis = self._parse_analysis(response, options)
        analysis['reasoning_type'] = 'chain_of_thought'
        analysis['token_info'] = token_info

        return analysis
