# src/agents/medical_agent.py
from typing import Dict, List, Optional, Tuple
from .base_agent import BaseAgent
import json


class MedicalAgent(BaseAgent):
    """Base class for all medical specialist agents"""

    def __init__(self, specialty: str, agent_id: str, expertise: str,
                 relevance_score: float = 1.0, decision_weight: float = None,
                 model: str = "gpt-4o", logger=None):

        # Create personalized instruction based on specialty
        instruction = f"""You are a {specialty} with expertise in {expertise}.
        
        When analyzing medical cases:
        1. Consider all symptoms and clinical findings carefully
        2. Apply your specialized knowledge to identify relevant patterns
        3. Provide differential diagnoses when appropriate
        4. Explain your reasoning clearly
        5. Be confident but acknowledge uncertainty when present
        
        Always think step-by-step and provide evidence-based recommendations."""

        super().__init__(
            role=specialty,
            instruction=instruction,
            model=model,
            logger=logger
        )

        self.agent_id = agent_id
        self.specialty = specialty
        self.expertise = expertise
        self.relevance_score = relevance_score
        self.decision_weight = decision_weight
        self.current_case = None

    def analyze_case(self, question: str, options: Dict[str, str] = None) -> Dict:
        """Base method for analyzing a medical case"""

        # Format the question with options if provided
        if options:
            formatted_question = f"{question}\n\nOptions:\n"
            for key, value in sorted(options.items()):
                formatted_question += f"{key}) {value}\n"
        else:
            formatted_question = question

        prompt = f"""Medical Case Analysis:
        
{formatted_question}

Please provide:
1. Key clinical findings
2. Your differential diagnosis (if applicable)
3. Most likely answer with reasoning
4. Confidence level (high/medium/low)"""

        response, token_info = self.chat(prompt)

        # Parse the response
        analysis = self._parse_analysis(response, options)
        analysis['token_info'] = token_info

        return analysis

    def _parse_analysis(self, response: str, options: Dict[str, str] = None) -> Dict:
        """Parse the analysis response"""
        analysis = {
            "raw_response": response,
            "clinical_findings": "",
            "differential_diagnosis": "",
            "answer": "",
            "reasoning": "",
            "confidence": "medium"
        }

        # Extract answer if options provided
        if options:
            for key in options.keys():
                if f"{key})" in response or f"({key})" in response:
                    analysis["answer"] = key
                    break

        # Extract confidence level
        response_lower = response.lower()
        if "high confidence" in response_lower or "very confident" in response_lower:
            analysis["confidence"] = "high"
        elif "low confidence" in response_lower or "uncertain" in response_lower:
            analysis["confidence"] = "low"

        # Store full response as reasoning for now
        analysis["reasoning"] = response

        return analysis

    def discuss_with_colleague(self, colleague_opinion: str, colleague_specialty: str) -> str:
        """Respond to a colleague's opinion"""
        prompt = f"""A {colleague_specialty} colleague has shared their opinion:

{colleague_opinion}

As a {self.specialty}, please:
1. Consider their perspective
2. Agree or respectfully disagree with reasoning
3. Add any insights from your expertise
4. Suggest how to reconcile different views if applicable"""

        response, _ = self.chat(prompt, temperature=0.5)
        return response

    def coordinate_silent_assessment_phase(self, question: str, options: Dict[str, str]) -> Dict:
        """Phase 1: Collect silent SBAR assessments from all team members"""
        print("\n--- Collecting Silent SBAR Assessments ---")

        # Get lead's assessment
        lead_assessment = self.generate_sbar_assessment(question, options)
        self.silent_assessments[self.agent_id] = lead_assessment

        print(f"\n✓ Lead ({self.specialty}) completed assessment")

        # Get team member assessments
        for member in self.team_members:
            if hasattr(member, 'generate_sbar_assessment'):
                assessment = member.generate_sbar_assessment(question, options)
                self.silent_assessments[member.agent_id] = assessment
                print(f"✓ {member.specialty} completed assessment")

        print(
            f"\n✓ Silent assessment phase complete ({len(self.silent_assessments)} assessments)")

        return self.silent_assessments

    def generate_sbar_assessment(self, question: str, options: Dict[str, str]) -> Dict:
        """Generate SBAR-style assessment"""

        # Format the case
        formatted_question = f"{question}\n\nOptions:\n"
        for key, value in sorted(options.items()):
            formatted_question += f"{key}) {value}\n"

        sbar_prompt = f"""As a {self.specialty}, provide a structured SBAR assessment for this case:

    {formatted_question}

    Please structure your response as:

    SITUATION: (Brief statement of the problem)
    BACKGROUND: (Relevant clinical context from your specialty perspective)  
    ASSESSMENT: (Your clinical assessment and most likely diagnosis)
    RECOMMENDATION: (Your recommended answer and key reasoning)

    Keep it concise (under 200 words total)."""

        response, token_info = self.chat(sbar_prompt, temperature=0.7)

        # Parse SBAR components
        sbar_assessment = self._parse_sbar(response, options)
        sbar_assessment['token_info'] = token_info

        return sbar_assessment

    def _parse_sbar(self, response: str, options: Dict[str, str]) -> Dict:
        """Parse SBAR format response"""
        sbar = {
            "situation": "",
            "background": "",
            "assessment": "",
            "recommendation": "",
            "recommended_answer": None,
            "raw_response": response
        }

        # Extract sections
        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_upper = line.upper()
            if "SITUATION:" in line_upper:
                current_section = "situation"
                sbar[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "BACKGROUND:" in line_upper:
                current_section = "background"
                sbar[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "ASSESSMENT:" in line_upper:
                current_section = "assessment"
                sbar[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "RECOMMENDATION:" in line_upper:
                current_section = "recommendation"
                sbar[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif current_section:
                sbar[current_section] += " " + line.strip()

        # Extract recommended answer
        for key in options.keys():
            if f"{key})" in response or f"({key})" in response or f"Answer: {key}" in response:
                sbar["recommended_answer"] = key
                break

        return sbar
