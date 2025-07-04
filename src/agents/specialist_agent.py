# src/agents/specialist_agent.py
from typing import Dict, List, Optional, Tuple
from .medical_agent import MedicalAgent
import json


class SpecialistAgent(MedicalAgent):
    """Specialist agent for moderate complexity collaborative cases"""

    def __init__(self, specialty: str, agent_id: str, expertise: str,
                 relevance_score: float = 1.0, decision_weight: float = 0.2,
                 logger=None):
        super().__init__(
            specialty=specialty,
            agent_id=agent_id,
            expertise=expertise,
            relevance_score=relevance_score,
            decision_weight=decision_weight,
            logger=logger
        )

        self.silent_assessment = None
        self.vote = None
        self.discussion_history = []

    def generate_sbar_assessment(self, question: str, options: Dict[str, str]) -> Dict:
        """Generate SBAR-style assessment for silent pre-round input"""

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
        self.silent_assessment = sbar_assessment

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

    def participate_in_round_robin(self, previous_opinions: List[Dict]) -> str:
        """Share opinion during structured round-robin discussion"""

        # Build context from previous speakers
        context = "Previous specialists have shared:\n\n"
        for opinion in previous_opinions[-3:]:  # Last 3 opinions for context
            context += f"{opinion['specialty']}: {opinion['summary']}\n"

        prompt = f"""As a {self.specialty}, it's your turn in the round-robin discussion.

{context}

Based on your SBAR assessment:
- {self.silent_assessment['assessment']}
- Recommendation: {self.silent_assessment['recommendation']}

Please share your perspective concisely (2-3 sentences max). 
Focus on insights unique to your specialty."""

        response, _ = self.chat(prompt, temperature=0.7)
        return response

    def participate_in_open_discussion(self, discussion_point: str,
                                       other_specialist: str = None) -> str:
        """Participate in open discussion phase"""

        prompt = f"""Open discussion point: {discussion_point}"""

        if other_specialist:
            prompt += f"\n\nThis was raised by the {other_specialist}."

        prompt += f"""

As a {self.specialty}, provide your input on this point.
Be constructive and focus on patient care.
Keep response under 100 words."""

        response, _ = self.chat(prompt, temperature=0.7)
        self.discussion_history.append({
            "point": discussion_point,
            "response": response
        })

        return response

    def cast_delphi_vote(self, options: Dict[str, str],
                         discussion_summary: str = None) -> Dict:
        """Cast vote in Delphi-lite consensus process"""

        vote_prompt = f"""After the team discussion, please cast your vote.

Options:
"""
        for key, value in sorted(options.items()):
            vote_prompt += f"{key}) {value}\n"

        if discussion_summary:
            vote_prompt += f"\nDiscussion Summary:\n{discussion_summary}\n"

        vote_prompt += f"""
As a {self.specialty}, please provide:
1. Your vote (letter only)
2. Confidence (0.0-1.0)
3. Brief rationale (one sentence)

Format: 
VOTE: [letter]
CONFIDENCE: [number]
RATIONALE: [text]"""

        # Lower temp for consistency
        response, _ = self.chat(vote_prompt, temperature=0.3)

        # Parse vote
        vote_data = self._parse_vote(response)
        self.vote = vote_data

        return vote_data

    def _parse_vote(self, response: str) -> Dict:
        """Parse voting response"""
        vote_data = {
            "choice": None,
            "confidence": 0.5,
            "rationale": "",
            "raw_response": response
        }

        lines = response.split('\n')
        for line in lines:
            line_upper = line.upper()
            if "VOTE:" in line_upper:
                # Extract letter
                for char in line:
                    if char.upper() in ['A', 'B', 'C', 'D', 'E']:
                        vote_data["choice"] = char.upper()
                        break
            elif "CONFIDENCE:" in line_upper:
                # Extract confidence
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    vote_data["confidence"] = float(conf_str)
                except:
                    pass
            elif "RATIONALE:" in line_upper:
                vote_data["rationale"] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""

        return vote_data

    def acknowledge_final_decision(self, final_decision: str,
                                   decision_rationale: str) -> str:
        """Acknowledge the final decision made by lead physician"""

        prompt = f"""The lead physician has made the final decision: {final_decision}

Rationale: {decision_rationale}

As a {self.specialty}, briefly acknowledge this decision and note any important 
follow-up considerations from your specialty perspective (if any).
Keep response under 50 words."""

        response, _ = self.chat(prompt, temperature=0.5)
        return response
