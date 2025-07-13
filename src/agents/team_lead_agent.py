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
        self.discussion_logs = []  # Changed to store multiple rounds
        self.voting_results = {}
        self.final_decision = None
        self.interaction_history = []  # New: stores all interactions
        self.consensus_threshold = 0.8  # 80% agreement

    def set_team_members(self, team_members: List):
        """Set the team members this lead will coordinate"""
        self.team_members = team_members

    def coordinate_moderate_complexity_case(self, question: str, options: Dict[str, str],
                                            max_rounds: int = 5) -> Dict:
        """Main coordination method following Algorithm 1 (Lines 6-22)"""

        print("\n" + "="*80)
        print("MODERATE COMPLEXITY CASE - MDT COORDINATION")
        print("="*80)

        # Initialize
        r = 0
        consensus = False
        self.interaction_history = []

        # Phase 1: Silent Assessment
        print("\n>>> Phase 1: Silent Pre-Round Assessment")
        self.coordinate_silent_assessment_phase(question, options)

        # Check initial consensus from silent assessments
        initial_consensus = self.check_initial_consensus()
        if initial_consensus['has_consensus']:
            print(
                f"\n✓ Initial consensus reached: Option {initial_consensus['choice']}")
            return self.make_final_decision(options)

        # Iterative consensus loop (Algorithm 1, lines 11-21)
        while r < max_rounds and not consensus:
            print(f"\n" + "="*60)
            print(f"ROUND {r+1} of {max_rounds}")
            print("="*60)

            # Collaborative Discussion (Line 12)
            discussion_log = self.conduct_collaborative_discussion(
                r, question, options)
            self.discussion_logs.append(discussion_log)

            # Check consensus (Line 13)
            consensus_result = self.check_consensus()
            consensus = consensus_result['has_consensus']

            if not consensus and r < max_rounds - 1:
                print(
                    f"\n⚠ No consensus reached (agreement: {consensus_result['agreement_rate']:.1%})")

                # Moderator feedback (Lines 14-16)
                feedback = self.generate_moderator_feedback(
                    consensus_result, r)

                # Update agents with feedback (Line 16)
                self.update_agents_with_feedback(feedback)

                # Update interaction history (Line 18)
                self.interaction_history.extend(discussion_log)
                self.interaction_history.append({
                    'type': 'moderator_feedback',
                    'round': r + 1,
                    'feedback': feedback
                })
            elif consensus:
                print(
                    f"\n✓ Consensus reached: Option {consensus_result['majority_choice']}")

            r += 1

        # Make final decision (Line 22)
        return self.make_final_decision_with_interaction(options)

    def check_initial_consensus(self) -> Dict:
        """Check if there's consensus from silent assessments"""
        print("\nDEBUG: Checking initial consensus...")

        recommendations = []
        assessment_details = []

        for agent_id, assessment in self.silent_assessments.items():
            print(
                f"DEBUG: Agent {agent_id} assessment keys: {assessment.keys()}")

            if 'recommended_answer' in assessment and assessment['recommended_answer']:
                rec_answer = assessment['recommended_answer']
                recommendations.append(rec_answer)
                assessment_details.append((agent_id, rec_answer))
                print(f"DEBUG: {agent_id} recommends: {rec_answer}")
            else:
                print(f"DEBUG: {agent_id} has no clear recommendation")

        print(f"DEBUG: All recommendations: {recommendations}")

        if not recommendations:
            print("DEBUG: No recommendations found")
            return {'has_consensus': False, 'choice': None}

        # Count votes
        from collections import Counter
        vote_counts = Counter(recommendations)
        total_votes = len(recommendations)

        print(f"DEBUG: Vote counts: {vote_counts}")
        print(f"DEBUG: Total votes: {total_votes}")

        # Check if any option has >80% agreement
        for choice, count in vote_counts.items():
            agreement_rate = count / total_votes
            print(
                f"DEBUG: Option {choice}: {count}/{total_votes} = {agreement_rate:.1%}")

            if agreement_rate >= self.consensus_threshold:
                print(
                    f"DEBUG: Consensus found on {choice} with {agreement_rate:.1%} agreement")
                return {
                    'has_consensus': True,
                    'choice': choice,
                    'agreement_rate': agreement_rate,
                    'vote_breakdown': dict(vote_counts)
                }

        # If no consensus, return the majority choice anyway for testing
        if vote_counts:
            majority_choice = vote_counts.most_common(1)[0][0]
            majority_rate = vote_counts[majority_choice] / total_votes
            print(
                f"DEBUG: No consensus, but majority is {majority_choice} with {majority_rate:.1%}")

            return {
                'has_consensus': False,
                'choice': majority_choice,
                'agreement_rate': majority_rate,
                'vote_breakdown': dict(vote_counts)
            }

        return {'has_consensus': False, 'choice': None}

    def conduct_collaborative_discussion(self, round_num: int, question: str,
                                         options: Dict[str, str]) -> List[Dict]:
        """Conduct collaborative discussion for current round"""

        discussion_log = []

        # Different discussion format based on round
        if round_num == 0:
            # First round: Structured round-robin
            print("\n--- Round 1: Structured Round-Robin ---")
            discussion_log = self._conduct_round_robin()

        elif round_num == 1:
            # Second round: Open discussion on key disagreements
            print("\n--- Round 2: Focused Discussion on Disagreements ---")
            discussion_log = self._conduct_focused_discussion(options)

        else:
            # Later rounds: Targeted problem-solving
            print(
                f"\n--- Round {round_num + 1}: Problem-Solving Discussion ---")
            discussion_log = self._conduct_problem_solving_discussion(options)

        return discussion_log

    def _conduct_round_robin(self) -> List[Dict]:
        """Standard round-robin discussion (already implemented)"""
        # Use existing conduct_round_robin_discussion method
        return self.conduct_round_robin_discussion()

    def _conduct_focused_discussion(self, options: Dict[str, str]) -> List[Dict]:
        """Focus on specific disagreements"""
        discussion_log = []

        # Identify main disagreements
        disagreements = self._identify_disagreements()

        for disagreement in disagreements[:2]:  # Focus on top 2 disagreements
            print(f"\nDiscussing: {disagreement['issue']}")

            # Get input from agents with different views
            for agent in disagreement['agents_involved']:
                response = agent.participate_in_open_discussion(
                    f"Please clarify your position on: {disagreement['issue']}",
                    "Moderator"
                )

                discussion_log.append({
                    'speaker': agent,
                    'specialty': agent.specialty,
                    'topic': disagreement['issue'],
                    'message': response
                })

                print(f"  {agent.specialty}: {response[:100]}...")

        return discussion_log

    def _conduct_problem_solving_discussion(self, options: Dict[str, str]) -> List[Dict]:
        """Problem-solving approach for persistent disagreements"""
        discussion_log = []

        # Identify the main contention points
        current_votes = self._get_current_preferences()

        problem_prompt = f"""The team has not reached consensus after multiple rounds.
        Current preferences: {current_votes}
        
        Please provide a brief statement on:
        1. What additional information would help resolve this?
        2. Any compromise position you would consider?"""

        for member in self.team_members:
            response = member.participate_in_open_discussion(
                problem_prompt, "Moderator")

            discussion_log.append({
                'speaker': member,
                'specialty': member.specialty,
                'message': response
            })

            print(f"\n{member.specialty}: {response[:150]}...")

        return discussion_log

    def check_consensus(self) -> Dict:
        """Check if consensus has been reached (>80% agreement)"""

        # Conduct quick poll
        print("\n--- Checking Consensus ---")
        current_votes = {}

        # Lead votes
        lead_choice = self._get_lead_preference()
        current_votes[self.agent_id] = lead_choice

        # Team member votes
        for member in self.team_members:
            if hasattr(member, 'get_current_preference'):
                choice = member.get_current_preference()
            else:
                # Quick vote based on current discussion
                choice = self._quick_poll_member(member)
            current_votes[member.agent_id] = choice

        # Analyze votes
        vote_counts = Counter(current_votes.values())
        total_votes = len(current_votes)

        # Find majority choice and agreement rate
        if vote_counts:
            majority_choice, majority_count = vote_counts.most_common(1)[0]
            agreement_rate = majority_count / total_votes

            return {
                'has_consensus': agreement_rate >= self.consensus_threshold,
                'agreement_rate': agreement_rate,
                'majority_choice': majority_choice,
                'vote_distribution': dict(vote_counts),
                'detailed_votes': current_votes
            }

        return {
            'has_consensus': False,
            'agreement_rate': 0,
            'majority_choice': None
        }

    def generate_moderator_feedback(self, consensus_result: Dict, round_num: int) -> Dict:
        """Generate targeted feedback when no consensus is reached"""

        feedback_prompt = f"""As moderator, the team has not reached consensus after round {round_num + 1}.

Vote distribution: {consensus_result['vote_distribution']}
Agreement rate: {consensus_result['agreement_rate']:.1%}

Generate brief, targeted feedback to help the team converge:
1. Identify the key sticking point
2. Suggest what perspective might be missing
3. Propose a specific question for the team to consider

Keep total feedback under 150 words."""

        response, _ = self.chat(feedback_prompt, temperature=0.6)

        feedback = {
            'round': round_num + 1,
            'content': response,
            'vote_distribution': consensus_result['vote_distribution'],
            'target_areas': self._identify_feedback_targets(consensus_result)
        }

        print(f"\n📋 Moderator Feedback:")
        print(response)

        return feedback

    def update_agents_with_feedback(self, feedback: Dict):
        """Update agents with moderator feedback"""

        print("\n--- Updating Agents with Feedback ---")

        for member in self.team_members:
            if hasattr(member, 'receive_feedback'):
                member.receive_feedback(feedback['content'])
            else:
                # Create a simple update message
                update_prompt = f"""The moderator has provided feedback:

{feedback['content']}

Please consider this feedback for the next discussion round."""

                # Store in agent's discussion history
                member.discussion_history.append({
                    'type': 'moderator_feedback',
                    'content': update_prompt
                })

        print("✓ All agents updated with moderator feedback")

    def make_final_decision_with_interaction(self, options: Dict[str, str]) -> Dict:
        """Make final decision considering all interaction history"""

        print("\n" + "="*60)
        print("FINAL DECISION PHASE")
        print("="*60)

        # Get final consensus state
        final_consensus = self.check_consensus()

        # Create interaction summary
        interaction_summary = self._create_interaction_summary()

        decision_prompt = f"""As Team Lead, make the final decision after the MDT process.

Process Summary:
- Total rounds: {len(self.discussion_logs)}
- Final consensus: {'Yes' if final_consensus['has_consensus'] else 'No'}
- Agreement rate: {final_consensus['agreement_rate']:.1%}
- Vote distribution: {final_consensus['vote_distribution']}

Interaction highlights:
{interaction_summary}

Options:
"""
        for key, value in sorted(options.items()):
            decision_prompt += f"{key}) {value}\n"

        decision_prompt += """
Provide:
DECISION: [letter]
PRIMARY RATIONALE: [2-3 sentences explaining your decision]
MINORITY CONSIDERATION: [acknowledge any dissenting views]
CONSENSUS STATUS: [was consensus achieved? If not, why did you choose this option?]
FOLLOW-UP: [any monitoring or contingency plans needed]"""

        response, token_info = self.chat(decision_prompt, temperature=0.5)

        # Parse decision
        decision = self._parse_final_decision(response, options)
        decision['consensus_achieved'] = final_consensus['has_consensus']
        decision['final_agreement_rate'] = final_consensus['agreement_rate']
        decision['token_info'] = token_info

        self.final_decision = decision

        # Notify team
        self._notify_team_of_decision(decision)

        print(f"\n✓ FINAL DECISION: Option {decision['choice']}")
        print(f"Consensus achieved: {decision['consensus_achieved']}")
        print(f"Agreement rate: {decision['final_agreement_rate']:.1%}")

        return decision

    # Helper methods
    def _identify_disagreements(self) -> List[Dict]:
        """Identify key disagreements from assessments and discussions"""
        disagreements = []

        # Analyze silent assessments
        answer_groups = {}
        for agent_id, assessment in self.silent_assessments.items():
            if 'recommended_answer' in assessment:
                answer = assessment['recommended_answer']
                if answer not in answer_groups:
                    answer_groups[answer] = []
                answer_groups[answer].append(agent_id)

        # Find conflicting groups
        if len(answer_groups) > 1:
            sorted_groups = sorted(answer_groups.items(),
                                   key=lambda x: len(x[1]), reverse=True)

            disagreements.append({
                'issue': f"Choice between option {sorted_groups[0][0]} vs {sorted_groups[1][0]}",
                'agents_involved': [self._get_agent_by_id(aid) for aid in
                                    sorted_groups[0][1] + sorted_groups[1][1]]
            })

        return disagreements

    def _get_current_preferences(self) -> Dict:
        """Get current voting preferences"""
        preferences = {}
        for member in self.team_members:
            if hasattr(member, 'vote') and member.vote:
                preferences[member.specialty] = member.vote.get('choice', '?')
        return preferences

    def _get_lead_preference(self) -> str:
        """Get lead's current preference"""
        # Simplified - in practice would analyze discussion
        if self.silent_assessments.get(self.agent_id):
            return self.silent_assessments[self.agent_id].get('recommended_answer', 'A')
        return 'A'

    def _quick_poll_member(self, member) -> str:
        """Quick poll for current preference"""
        # Simplified - in practice would be more sophisticated
        if hasattr(member, 'silent_assessment') and member.silent_assessment:
            return member.silent_assessment.get('recommended_answer', 'A')
        return 'A'

    def _identify_feedback_targets(self, consensus_result: Dict) -> List[str]:
        """Identify specific areas needing feedback"""
        targets = []

        vote_dist = consensus_result['vote_distribution']
        if len(vote_dist) > 2:
            targets.append("Multiple divergent opinions")
        elif len(vote_dist) == 2:
            targets.append("Binary disagreement")

        return targets

    def _create_interaction_summary(self) -> str:
        """Create summary of interaction history"""
        summary_points = []

        summary_points.append(
            f"- {len(self.discussion_logs)} rounds of discussion conducted")

        if self.interaction_history:
            feedback_count = sum(1 for i in self.interaction_history
                                 if i.get('type') == 'moderator_feedback')
            if feedback_count > 0:
                summary_points.append(
                    f"- {feedback_count} moderator interventions")

        if hasattr(self, 'voting_results') and self.voting_results:
            summary_points.append(
                f"- Formal voting conducted with {len(self.voting_results)} participants")

        return "\n".join(summary_points)

    def _get_agent_by_id(self, agent_id: str):
        """Get agent instance by ID"""
        if agent_id == self.agent_id:
            return self
        for member in self.team_members:
            if member.agent_id == agent_id:
                return member
        return None

    # Update existing methods to work with new structure
    def coordinate_silent_assessment_phase(self, question: str, options: Dict[str, str]) -> Dict:
        """Phase 1: Collect silent SBAR assessments from all team members"""
        # (Keep existing implementation)
        return super().coordinate_silent_assessment_phase(question, options)

    def _parse_final_decision(self, response: str, options: Dict[str, str]) -> Dict:
        """Parse the final decision with consensus status"""
        decision = {
            "choice": None,
            "rationale": "",
            "minority_consideration": "",
            "consensus_status": "",
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
            elif "CONSENSUS STATUS:" in line_upper:
                current_section = "consensus_status"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "FOLLOW-UP:" in line_upper:
                current_section = "follow_up"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif current_section and line.strip():
                decision[current_section] += " " + line.strip()

        return decision

# Add these methods to your TeamLeadAgent class

    def make_final_decision(self, options: Dict[str, str]) -> Dict:
        """Make final decision when early consensus is reached"""

        print("\n" + "="*60)
        print("EARLY CONSENSUS ACHIEVED")
        print("="*60)

        # Get the consensus choice from initial assessments
        initial_consensus = self.check_initial_consensus()
        consensus_choice = initial_consensus['choice']
        agreement_rate = initial_consensus['agreement_rate']

        print(f"DEBUG: Consensus choice = {consensus_choice}")
        print(f"DEBUG: Agreement rate = {agreement_rate:.1%}")

        # Simplified decision - just use the consensus choice directly
        decision = {
            "choice": consensus_choice,  # Use consensus choice directly
            "rationale": f"Team reached {agreement_rate:.1%} consensus on option {consensus_choice} during silent assessment phase.",
            "team_summary": f"All {len(self.team_members)} specialists agreed on this diagnosis.",
            "confidence": "High - based on unanimous team agreement",
            "consensus_achieved": True,
            "early_consensus": True,
            "final_agreement_rate": agreement_rate,
            "raw_response": f"Early consensus decision: {consensus_choice}"
        }

        self.final_decision = decision

        print(f"\n✓ EARLY CONSENSUS DECISION: Option {decision['choice']}")
        print(f"Agreement rate: {agreement_rate:.1%}")

        return decision

    def _parse_early_consensus_decision(self, response: str, options: Dict[str, str],
                                        expected_choice: str) -> Dict:
        """Parse early consensus decision response"""
        decision = {
            "choice": expected_choice,  # Default to consensus choice
            "rationale": "",
            "team_summary": "",
            "confidence": "",
            "raw_response": response
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_upper = line.upper()

            if "DECISION:" in line_upper:
                # Extract choice, but default to expected if not found
                for char in line:
                    if char.upper() in options.keys():
                        decision["choice"] = char.upper()
                        break
            elif "RATIONALE:" in line_upper:
                current_section = "rationale"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "TEAM_SUMMARY:" in line_upper:
                current_section = "team_summary"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif "CONFIDENCE:" in line_upper:
                current_section = "confidence"
                decision[current_section] = line.split(
                    ':', 1)[1].strip() if ':' in line else ""
            elif current_section and line.strip():
                decision[current_section] += " " + line.strip()

        return decision

    def conduct_round_robin_discussion(self) -> List[Dict]:
        """Conduct structured round-robin discussion"""
        discussion_log = []

        print("\n--- Round-Robin Discussion ---")

        # Each team member shares their perspective in order
        previous_opinions = []

        for i, member in enumerate(self.team_members):
            print(f"\n{i+1}. {member.specialty} sharing perspective...")

            if hasattr(member, 'participate_in_round_robin'):
                response = member.participate_in_round_robin(previous_opinions)
            else:
                # Fallback for members without this method
                response = self._get_member_opinion(member)

            # Create opinion summary
            opinion = {
                'specialty': member.specialty,
                'summary': response[:200] + "..." if len(response) > 200 else response,
                'full_response': response
            }

            previous_opinions.append(opinion)

            discussion_log.append({
                'speaker': member,
                'specialty': member.specialty,
                'message': response,
                'order': i + 1
            })

            print(f"   {member.specialty}: {response[:100]}...")

        return discussion_log

    def _get_member_opinion(self, member) -> str:
        """Fallback method to get member opinion"""
        if hasattr(member, 'silent_assessment') and member.silent_assessment:
            assessment = member.silent_assessment.get('assessment', '')
            recommendation = member.silent_assessment.get('recommendation', '')
            return f"Assessment: {assessment} Recommendation: {recommendation}"
        else:
            return f"As a {member.specialty}, I believe we should consider the clinical evidence carefully."

    def _notify_team_of_decision(self, decision: Dict):
        """Notify team members of final decision"""
        print("\n--- Notifying Team of Final Decision ---")

        decision_summary = f"Final Decision: Option {decision['choice']}"
        rationale = decision.get('rationale', 'See discussion summary')

        for member in self.team_members:
            if hasattr(member, 'acknowledge_final_decision'):
                response = member.acknowledge_final_decision(
                    decision_summary, rationale)
                print(f"✓ {member.specialty}: {response[:80]}...")
            else:
                print(f"✓ {member.specialty}: Acknowledged decision")

    # Also add this missing method to the base MedicalAgent class if needed
    def conduct_round_robin_discussion(self) -> List[Dict]:
        """Base implementation for round-robin discussion"""
        return []

    def _parse_sbar(self, response: str, options: Dict[str, str]) -> Dict:
        """Parse SBAR format response with better answer extraction"""
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

        # Better answer extraction - try multiple patterns
        answer_patterns = [
            r'Answer:\s*\(?([A-E])\)?',  # Answer: A or Answer: (A)
            r'Option\s*([A-E])',         # Option A
            r'\(([A-E])\)',              # (A)
            r'^([A-E])\)',               # A) at start of line
            r'choose\s*([A-E])',         # choose A
            r'select\s*([A-E])',         # select A
        ]

        import re
        response_upper = response.upper()

        for pattern in answer_patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                # Take the last match (most likely to be the final answer)
                candidate = matches[-1]
                if candidate in options.keys():
                    sbar["recommended_answer"] = candidate
                    print(
                        f"DEBUG: Extracted answer '{candidate}' using pattern {pattern}")
                    break

        # Fallback: look for any option letter in the response
        if not sbar["recommended_answer"]:
            option_counts = {}
            for key in options.keys():
                option_counts[key] = response_upper.count(key)

            if option_counts:
                # Choose the most frequently mentioned option
                most_mentioned = max(option_counts, key=option_counts.get)
                if option_counts[most_mentioned] > 0:
                    sbar["recommended_answer"] = most_mentioned
                    print(
                        f"DEBUG: Fallback answer '{most_mentioned}' (mentioned {option_counts[most_mentioned]} times)")

        print(f"DEBUG: Final recommended answer: {sbar['recommended_answer']}")
        return sbar
