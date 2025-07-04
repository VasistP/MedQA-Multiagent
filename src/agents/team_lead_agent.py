# src/agents/team_lead_agent.py
from typing import Dict, List, Optional, Tuple
from .medical_agent import MedicalAgent
import json
from collections import Counter


class TeamLeadAgent(MedicalAgent):
    """Team Lead physician for coordinating moderate complexity cases"""

    def __init__(self, agent_id: str = "lead_physician", logger=None):
        super().__init__(
            specialty="Internal Medicine - Team Lead",
            agent_id=agent_id,
            expertise="comprehensive medical knowledge, clinical leadership, team coordination",
            relevance_score=0.9,
            decision_weight=0.3,  # Lead has 30% weight
            logger=logger
        )

        self.team_members = []
        self.silent_assessments = {}
        self.discussion_log = []
        self.voting_results = {}
        self.final_decision = None

    def set_team_members(self, team_members: List):
        """Set the team members this lead will coordinate"""
        self.team_members = team_members

    def coordinate_silent_assessment_phase(self, question: str, options: Dict[str, str]) -> Dict:
        """Phase 1: Collect silent SBAR assessments from all team members"""

        print("\n=== PHASE 1: Silent Pre-Round Assessment ===")
        self.silent_assessments = {}

        # First, lead does their own assessment
        print(f"\n{self.specialty} preparing assessment...")
        lead_assessment = self._generate_lead_assessment(question, options)
        self.silent_assessments[self.agent_id] = lead_assessment

        # Collect from team members
        for member in self.team_members:
            if hasattr(member, 'generate_sbar_assessment'):
                print(f"\n{member.specialty} preparing SBAR assessment...")
                assessment = member.generate_sbar_assessment(question, options)
                self.silent_assessments[member.agent_id] = assessment

        # Log phase completion
        if self.logger:
            self.logger.log_discussion(
                round_num=1,
                turn_num=0,
                agent_from="System",
                agent_to="All",
                message=f"Silent assessment phase completed. {len(self.silent_assessments)} assessments collected."
            )

        return self.silent_assessments

    def _generate_lead_assessment(self, question: str, options: Dict[str, str]) -> Dict:
        """Generate lead physician's assessment"""
        formatted_question = f"{question}\n\nOptions:\n"
        for key, value in sorted(options.items()):
            formatted_question += f"{key}) {value}\n"

        prompt = f"""As the Team Lead physician, provide your initial assessment:

{formatted_question}

Provide a brief clinical assessment focusing on:
1. Key differential diagnoses
2. Critical findings to consider
3. Initial recommendation

Keep it concise but comprehensive."""

        response, token_info = self.chat(prompt, temperature=0.7)

        return {
            "assessment": response,
            "specialty": self.specialty,
            "token_info": token_info
        }

    def conduct_round_robin_discussion(self) -> List[Dict]:
        """Phase 2: Structured round-robin discussion"""

        print("\n=== PHASE 2: Structured Round-Robin Discussion ===")
        discussion_sequence = []

        # Determine order based on specialties
        ordered_members = self._determine_discussion_order()

        # Conduct round-robin
        for i, member in enumerate(ordered_members):
            print(f"\n[Round-Robin Turn {i+1}] {member.specialty}:")

            # Get previous opinions for context
            previous_opinions = [
                {
                    "specialty": op["speaker"].specialty,
                    "summary": op["message"][:100] + "..."
                }
                for op in discussion_sequence
            ]

            # Get member's input
            if hasattr(member, 'participate_in_round_robin'):
                opinion = member.participate_in_round_robin(previous_opinions)
            else:
                opinion = "Lead physician notes all assessments for consideration."

            discussion_entry = {
                "speaker": member,
                "specialty": member.specialty,
                "message": opinion,
                "turn": i + 1
            }
            discussion_sequence.append(discussion_entry)

            print(f"{opinion[:200]}...")

            # Log discussion
            if self.logger:
                self.logger.log_discussion(
                    round_num=1,
                    turn_num=i+1,
                    agent_from=member.specialty,
                    agent_to="Team",
                    message=opinion
                )

        self.discussion_log = discussion_sequence
        return discussion_sequence

    def _determine_discussion_order(self) -> List:
        """Determine optimal order for round-robin based on specialties"""

        # Categories for ordering
        diagnostic_specialists = []
        organ_specialists = []
        support_specialists = []

        for member in self.team_members:
            specialty_lower = member.specialty.lower()

            if any(term in specialty_lower for term in ['pathologist', 'radiologist']):
                diagnostic_specialists.append(member)
            elif any(term in specialty_lower for term in ['pharmacist', 'social']):
                support_specialists.append(member)
            else:
                organ_specialists.append(member)

        # Order: Diagnostic → Organ (by relevance) → Support → Lead
        ordered = []
        ordered.extend(diagnostic_specialists)

        # Sort organ specialists by relevance score
        organ_specialists.sort(key=lambda x: x.relevance_score, reverse=True)
        ordered.extend(organ_specialists)

        ordered.extend(support_specialists)
        ordered.append(self)  # Lead goes last to summarize

        return ordered

    def facilitate_open_discussion(self, key_points: List[str] = None) -> List[Dict]:
        """Phase 3: Open discussion on key points"""

        print("\n=== PHASE 3: Open Discussion ===")
        open_discussions = []

        # Identify key discussion points if not provided
        if not key_points:
            key_points = self._identify_discussion_points()

        for point in key_points[:3]:  # Limit to top 3 points
            print(f"\nDiscussion Point: {point}")

            # Get responses from relevant members
            for member in self.team_members:
                if hasattr(member, 'participate_in_open_discussion'):
                    response = member.participate_in_open_discussion(point)
                    open_discussions.append({
                        "point": point,
                        "speaker": member.specialty,
                        "response": response
                    })
                    print(f"  {member.specialty}: {response[:100]}...")

        return open_discussions

    def _identify_discussion_points(self) -> List[str]:
        """Identify key points needing discussion"""

        # Analyze silent assessments for disagreements
        recommendations = []
        for assessment in self.silent_assessments.values():
            if 'recommended_answer' in assessment:
                recommendations.append(assessment['recommended_answer'])

        # Check for disagreement
        recommendation_counts = Counter(recommendations)

        points = []
        if len(recommendation_counts) > 1:
            points.append("Reconciling different diagnostic opinions")

        points.extend([
            "Risk-benefit analysis of the leading option",
            "Alternative management if first choice fails"
        ])

        return points

    def conduct_delphi_voting(self, options: Dict[str, str]) -> Dict:
        """Phase 4: Delphi-lite voting process"""

        print("\n=== PHASE 4: Delphi-Lite Voting ===")

        # Create discussion summary
        summary = self._create_discussion_summary()

        # Collect votes
        self.voting_results = {}

        # Lead votes too
        print(f"\n{self.specialty} casting vote...")
        lead_vote = self._cast_lead_vote(options, summary)
        self.voting_results[self.agent_id] = lead_vote

        # Collect team votes
        for member in self.team_members:
            if hasattr(member, 'cast_delphi_vote'):
                print(f"{member.specialty} casting vote...")
                vote = member.cast_delphi_vote(options, summary)
                self.voting_results[member.agent_id] = vote

        # Analyze results
        vote_analysis = self._analyze_votes()

        print("\n--- Voting Results ---")
        for choice, data in vote_analysis.items():
            print(f"Option {choice}: {data['count']} votes, "
                  f"avg confidence: {data['avg_confidence']:.2f}")

        return self.voting_results

    def _cast_lead_vote(self, options: Dict[str, str], summary: str) -> Dict:
        """Lead physician casts their vote"""

        vote_prompt = f"""As Team Lead, after hearing all perspectives, cast your vote:

Options:
"""
        for key, value in sorted(options.items()):
            vote_prompt += f"{key}) {value}\n"

        vote_prompt += f"""
Summary: {summary}

Provide:
VOTE: [letter]
CONFIDENCE: [0.0-1.0]
RATIONALE: [one sentence]"""

        response, _ = self.chat(vote_prompt, temperature=0.3)

        # Parse vote (similar to specialist parsing)
        vote_data = {
            "choice": None,
            "confidence": 0.7,
            "rationale": "Lead physician's clinical judgment",
            "specialty": self.specialty
        }

        # Extract vote details
        lines = response.split('\n')
        for line in lines:
            if "VOTE:" in line.upper():
                for char in line:
                    if char.upper() in options.keys():
                        vote_data["choice"] = char.upper()
                        break
            elif "CONFIDENCE:" in line.upper():
                try:
                    conf = float(line.split(':', 1)[1].strip())
                    vote_data["confidence"] = conf
                except:
                    pass
            elif "RATIONALE:" in line.upper():
                vote_data["rationale"] = line.split(':', 1)[1].strip()

        return vote_data

    def _create_discussion_summary(self) -> str:
        """Create summary of key discussion points"""

        summary_points = []

        # Key agreements
        if self.discussion_log:
            summary_points.append("Team discussed multiple perspectives")

        # Note any major disagreements
        if self.silent_assessments:
            recommendations = [a.get('recommended_answer', '')
                               for a in self.silent_assessments.values()]
            if len(set(recommendations)) > 1:
                summary_points.append(
                    "Initial assessments showed varied opinions")

        return "; ".join(summary_points)

    def _analyze_votes(self) -> Dict:
        """Analyze voting results"""

        vote_counts = Counter()
        confidence_by_choice = {}

        for vote in self.voting_results.values():
            choice = vote.get('choice')
            if choice:
                vote_counts[choice] += 1
                if choice not in confidence_by_choice:
                    confidence_by_choice[choice] = []
                confidence_by_choice[choice].append(
                    vote.get('confidence', 0.5))

        analysis = {}
        for choice, count in vote_counts.items():
            analysis[choice] = {
                'count': count,
                'avg_confidence': sum(confidence_by_choice[choice]) / len(confidence_by_choice[choice]),
                'votes': confidence_by_choice[choice]
            }

        return analysis

    def make_final_decision(self, options: Dict[str, str]) -> Dict:
        """Phase 5: Make final decision with full accountability"""

        print("\n=== PHASE 5: Final Decision ===")

        # Analyze all inputs
        vote_analysis = self._analyze_votes()

        # Identify majority and minority opinions
        sorted_choices = sorted(vote_analysis.items(),
                                key=lambda x: (
                                    x[1]['count'], x[1]['avg_confidence']),
                                reverse=True)

        decision_prompt = f"""As Team Lead, make the final decision based on:

1. Silent assessments from all specialists
2. Round-robin discussion insights  
3. Voting results: {sorted_choices}

Options:
"""
        for key, value in sorted(options.items()):
            decision_prompt += f"{key}) {value}\n"

        decision_prompt += """
Provide:
DECISION: [letter]
PRIMARY RATIONALE: [2-3 sentences explaining your decision]
MINORITY CONSIDERATION: [acknowledge any dissenting views and why they were overruled]
FOLLOW-UP: [any monitoring or contingency plans needed]"""

        response, token_info = self.chat(decision_prompt, temperature=0.5)

        # Parse decision
        decision = self._parse_final_decision(response, options)
        decision['token_info'] = token_info

        # Document decision
        self.final_decision = decision

        # Notify team members
        self._notify_team_of_decision(decision)

        print(f"\nFINAL DECISION: Option {decision['choice']}")
        print(f"Rationale: {decision['rationale']}")

        return decision

    def _parse_final_decision(self, response: str, options: Dict[str, str]) -> Dict:
        """Parse the final decision"""

        decision = {
            "choice": None,
            "rationale": "",
            "minority_consideration": "",
            "follow_up": "",
            "raw_response": response
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_upper = line.upper()

            if "DECISION:" in line_upper:
                for char in line:
                    if char.upper() in options.keys():
                        decision["choice"] = char.upper()
                        break
            elif "PRIMARY RATIONALE:" in line_upper:
                current_section = "rationale"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "MINORITY CONSIDERATION:" in line_upper:
                current_section = "minority_consideration"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "FOLLOW-UP:" in line_upper:
                current_section = "follow_up"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif current_section and line.strip():
                decision[current_section] += " " + line.strip()

        return decision

    def _notify_team_of_decision(self, decision: Dict):
        """Notify team members of final decision"""

        for member in self.team_members:
            if hasattr(member, 'acknowledge_final_decision'):
                acknowledgment = member.acknowledge_final_decision(
                    f"Option {decision['choice']}",
                    decision['rationale']
                )

                if self.logger:
                    self.logger.log_discussion(
                        round_num=1,
                        turn_num=99,  # Final turn
                        agent_from=member.specialty,
                        agent_to="Lead",
                        message=f"Acknowledged: {acknowledgment}"
                    )
