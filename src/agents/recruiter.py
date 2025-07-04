# src/agents/recruiter.py (updated version)
from typing import List, Dict, Tuple, Union
import random
from collections import Counter
from .base_agent import BaseAgent
from .medical_specialties import MedicalSpecialties


class RecruiterAgent(BaseAgent):
    """Agent responsible for recruiting appropriate medical experts based on complexity"""

    INSTRUCTION = """You are an experienced medical expert who recruits other specialists to solve medical cases.
    Based on the complexity and nature of the medical question, you will select appropriate specialists.
    
    Your goal is to assemble the most relevant team of experts who can collaboratively solve the given medical problem.
    Consider the specific symptoms, systems involved, and required expertise when selecting specialists."""

    def __init__(self, logger=None):
        super().__init__(
            role="Medical Recruiter",
            instruction=self.INSTRUCTION,
            logger=logger
        )

    def calculate_relevance_scores(self, question: str) -> Dict[str, float]:
        """Calculate relevance scores for each specialty based on question content"""
        question_lower = question.lower()
        scores = {}

        for specialty, info in MedicalSpecialties.SPECIALTIES.items():
            # Base keyword matching
            keyword_score = sum(
                2 for keyword in info["keywords"] if keyword in question_lower)

            # Bonus for exact specialty terms
            if specialty.lower() in question_lower:
                keyword_score += 5

            # Additional scoring for critical terms
            critical_terms = {
                "emergency": ["Emergency Medicine", "Trauma Surgeon"],
                "child": ["Pediatrician", "Pediatric Specialist"],
                "cancer": ["Oncologist", "Hematologist", "Radiologist"],
                "heart": ["Cardiologist", "Cardiac Surgeon"],
                "brain": ["Neurologist", "Neurosurgeon"],
                "infection": ["Infectious Disease", "Internal Medicine"]
            }

            for term, specialists in critical_terms.items():
                if term in question_lower and specialty in specialists:
                    keyword_score += 3

            if keyword_score > 0:
                scores[specialty] = keyword_score

        # Normalize scores
        max_score = max(scores.values()) if scores else 1
        normalized_scores = {k: v/max_score for k, v in scores.items()}

        return normalized_scores

    def recruit_for_low_complexity(self, question: str) -> List[Dict]:
        """Recruit a single PCP for low complexity cases"""
        return [{
            "agent_id": "pcp_1",
            "specialty": "Primary Care Physician",
            "role": "Primary Decision Maker",
            "expertise": MedicalSpecialties.get_specialist_description("Primary Care Physician"),
            "relevance_score": 1.0  # PCP always fully relevant for low complexity
        }]

    def recruit_for_moderate_complexity(self, question: str) -> List[Dict]:
        """Recruit an MDT with hierarchy and relevance-based weighting"""
        # Calculate relevance scores
        relevance_scores = self.calculate_relevance_scores(question)

        # Sort specialists by relevance
        sorted_specialists = sorted(relevance_scores.items(),
                                    key=lambda x: x[1], reverse=True)

        # Create team with lead physician
        team = [{
            "agent_id": "lead_physician",
            "specialty": "Internal Medicine",
            "role": "Team Lead",
            "expertise": MedicalSpecialties.get_specialist_description("Internal Medicine"),
            "relevance_score": 0.9,  # Lead has high base relevance
            "decision_weight": 0.3  # Lead has 30% weight in final decision
        }]

        # Select top specialists (allow duplicates if highly relevant)
        selected_specialists = []
        specialist_count = Counter()

        for specialty, score in sorted_specialists[:8]:  # Consider top 8
            # Allow up to 2 of the same specialty if score > 0.8
            if specialist_count[specialty] < 2 or score > 0.8:
                selected_specialists.append((specialty, score))
                specialist_count[specialty] += 1

                if len(selected_specialists) >= 4:  # Need 4 specialists
                    break

        # If not enough, fill with diverse specialists
        if len(selected_specialists) < 4:
            remaining = [(s, relevance_scores.get(s, 0.1))
                         for s in MedicalSpecialties.SPECIALTIES.keys()
                         if specialist_count[s] == 0]
            remaining.sort(key=lambda x: x[1], reverse=True)
            selected_specialists.extend(
                remaining[:4-len(selected_specialists)])

        # Calculate decision weights based on relevance
        total_relevance = sum(score for _, score in selected_specialists)

        # Add specialists to team
        for i, (specialty, score) in enumerate(selected_specialists[:4]):
            # Weight is proportional to relevance, remaining 70% after lead
            weight = (score / total_relevance) * \
                0.7 if total_relevance > 0 else 0.175

            team.append({
                "agent_id": f"specialist_{i+1}",
                "specialty": specialty,
                "role": "Consulting Specialist",
                "expertise": MedicalSpecialties.get_specialist_description(specialty),
                "relevance_score": score,
                "decision_weight": weight
            })

        return team

    def recruit_for_high_complexity(self, question: str) -> Dict[str, List[Dict]]:
        """Recruit an ICT with IAT, DET, and FRDT teams"""
        # Calculate relevance scores
        relevance_scores = self.calculate_relevance_scores(question)
        sorted_specialists = sorted(relevance_scores.items(),
                                    key=lambda x: x[1], reverse=True)

        # Get pool of relevant specialists (allow duplicates)
        specialist_pool = []
        specialist_count = Counter()

        for specialty, score in sorted_specialists:
            # For high complexity, allow up to 3 of same specialty if very relevant
            if specialist_count[specialty] < 3 or score > 0.7:
                specialist_pool.append((specialty, score))
                specialist_count[specialty] += 1

        teams = {
            "Initial Assessment Team (IAT)": [
                {
                    "agent_id": "iat_lead",
                    "specialty": "Emergency Medicine",
                    "role": "IAT Team Lead",
                    "expertise": MedicalSpecialties.get_specialist_description("Emergency Medicine"),
                    "relevance_score": relevance_scores.get("Emergency Medicine", 0.8)
                },
                {
                    "agent_id": "iat_member_1",
                    "specialty": specialist_pool[0][0] if specialist_pool else "Internal Medicine",
                    "role": "IAT Primary Specialist",
                    "expertise": MedicalSpecialties.get_specialist_description(
                        specialist_pool[0][0] if specialist_pool else "Internal Medicine"
                    ),
                    "relevance_score": specialist_pool[0][1] if specialist_pool else 0.7
                },
                {
                    "agent_id": "iat_member_2",
                    "specialty": specialist_pool[1][0] if len(specialist_pool) > 1 else "Primary Care Physician",
                    "role": "IAT Secondary Specialist",
                    "expertise": MedicalSpecialties.get_specialist_description(
                        specialist_pool[1][0] if len(
                            specialist_pool) > 1 else "Primary Care Physician"
                    ),
                    "relevance_score": specialist_pool[1][1] if len(specialist_pool) > 1 else 0.6
                }
            ],
            "Diagnostic Evidence Team (DET)": [
                {
                    "agent_id": "det_lead",
                    "specialty": "Pathologist" if "Pathologist" not in [s[0] for s in specialist_pool[:2]]
                                else specialist_pool[2][0] if len(specialist_pool) > 2 else "Pathologist",
                    "role": "DET Team Lead",
                    "expertise": MedicalSpecialties.get_specialist_description("Pathologist"),
                    "relevance_score": relevance_scores.get("Pathologist", 0.7)
                },
                {
                    "agent_id": "det_member_1",
                    "specialty": "Radiologist" if "Radiologist" not in [s[0] for s in specialist_pool[:3]]
                                else specialist_pool[3][0] if len(specialist_pool) > 3 else "Radiologist",
                    "role": "DET Imaging Specialist",
                    "expertise": MedicalSpecialties.get_specialist_description("Radiologist"),
                    "relevance_score": relevance_scores.get("Radiologist", 0.7)
                },
                {
                    "agent_id": "det_member_2",
                    "specialty": specialist_pool[4][0] if len(specialist_pool) > 4 else "Hematologist",
                    "role": "DET Laboratory Specialist",
                    "expertise": MedicalSpecialties.get_specialist_description(
                        specialist_pool[4][0] if len(
                            specialist_pool) > 4 else "Hematologist"
                    ),
                    "relevance_score": specialist_pool[4][1] if len(specialist_pool) > 4 else 0.6
                }
            ],
            "Final Review & Decision Team (FRDT)": [
                {
                    "agent_id": "frdt_lead",
                    "specialty": "Internal Medicine",
                    "role": "FRDT Senior Consultant",
                    "expertise": "comprehensive medical knowledge, clinical decision making, treatment planning",
                    "relevance_score": 0.9  # Senior consultant always highly relevant
                },
                {
                    "agent_id": "frdt_member_1",
                    # Most relevant specialist
                    "specialty": specialist_pool[0][0] if specialist_pool else "Oncologist",
                    "role": "FRDT Primary Specialist",
                    "expertise": MedicalSpecialties.get_specialist_description(
                        specialist_pool[0][0] if specialist_pool else "Oncologist"
                    ),
                    "relevance_score": specialist_pool[0][1] if specialist_pool else 0.7
                },
                {
                    "agent_id": "frdt_member_2",
                    "specialty": "Clinical Pharmacist",
                    "role": "FRDT Medication Specialist",
                    "expertise": "medication interactions, dosing, pharmaceutical care, treatment optimization",
                    "relevance_score": 0.8  # Pharmacist important for final decisions
                }
            ]
        }

        return teams

    def generate_recruitment_explanation(self, question: str, complexity: str,
                                         recruited_agents: Union[List[Dict], Dict[str, List[Dict]]]) -> str:
        """Generate explanation for recruitment choices"""

        # For moderate/low complexity
        if isinstance(recruited_agents, list):
            specialists = [(agent['specialty'], agent.get('relevance_score', 0))
                           for agent in recruited_agents]
            specialist_str = ", ".join([f"{s[0]} (relevance: {s[1]:.2f})"
                                       for s in specialists[:5]])
        elif isinstance(recruited_agents, dict):
            # For high complexity (teams)
            all_specialists = []
            for team in recruited_agents.values():
                all_specialists.extend([(agent['specialty'], agent.get('relevance_score', 0))
                                       for agent in team])
            specialist_str = f"{len(recruited_agents)} teams with {len(all_specialists)} total specialists"

        prompt = f"""Given this medical question: {question[:500]}...
        
        Complexity level: {complexity}
        Recruited: {specialist_str}
        
        Briefly explain why this team composition was chosen (2-3 sentences)."""

        response, _ = self.chat(prompt, temperature=0.5)
        return response
