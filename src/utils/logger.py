# src/utils/logger.py
import os
import json
import wandb
from datetime import datetime
from typing import Dict, Any
from .cost_tracker import CostTracker


class MDAgentsLogger:
    def __init__(self, log_dir: str, use_wandb: bool = True):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.current_case_log = None
        self.cost_tracker = CostTracker(log_dir)  # Add cost tracker

        # Create log subdirectories
        os.makedirs(f"{log_dir}/cases", exist_ok=True)
        os.makedirs(f"{log_dir}/discussions", exist_ok=True)

        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "mdagents-medical"),
                config={
                    "model": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
                    "dataset": "MedQA",
                    "timestamp": datetime.now().isoformat()
                }
            )

    def log_api_call(self, tokens_used: int, model: str, input_tokens: int,
                     output_tokens: int, complexity: str = None):
        """Track API usage and cost"""
        if self.current_case_log:
            self.current_case_log["api_calls"] += 1
            self.current_case_log["total_tokens"] += tokens_used

        # Track cost
        cost = self.cost_tracker.add_api_call(
            model, input_tokens, output_tokens, complexity
        )

        if self.use_wandb:
            wandb.log({
                "api_cost": cost,
                "cumulative_cost": self.cost_tracker.session_costs["total_cost"]
            })

    def start_case(self, case_id: str, question: Dict[str, Any], complexity: str):
        """Start logging a new case"""
        self.current_case_log = {
            "case_id": case_id,
            "question": question,
            "complexity": complexity,
            "timestamp": datetime.now().isoformat(),
            "discussions": [],
            "api_calls": 0,
            "total_tokens": 0
        }

    def log_discussion(self, round_num: int, turn_num: int, agent_from: str,
                       agent_to: str, message: str):
        """Log agent discussions"""
        if self.current_case_log:
            self.current_case_log["discussions"].append({
                "round": round_num,
                "turn": turn_num,
                "from": agent_from,
                "to": agent_to,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })

    def end_case(self, prediction: str, correct: str, decision_reasoning: str):
        """Finalize case logging"""
        if self.current_case_log:
            self.current_case_log["prediction"] = prediction
            self.current_case_log["correct_answer"] = correct
            self.current_case_log["is_correct"] = prediction == correct
            self.current_case_log["decision_reasoning"] = decision_reasoning

            # Save to file
            case_file = f"{self.log_dir}/cases/case_{self.current_case_log['case_id']}.json"
            with open(case_file, 'w') as f:
                json.dump(self.current_case_log, f, indent=2)

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "accuracy": 1.0 if self.current_case_log["is_correct"] else 0.0,
                    "complexity": self.current_case_log["complexity"],
                    "api_calls": self.current_case_log["api_calls"],
                    "total_tokens": self.current_case_log["total_tokens"],
                })

            self.current_case_log = None

    def log_experiment_summary(self, results: Dict[str, Any]):
        """Log overall experiment results"""
        summary_file = f"{self.log_dir}/experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.use_wandb:
            wandb.summary.update(results)
