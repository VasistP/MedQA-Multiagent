# src/utils/cost_tracker.py
import json
import os
from datetime import datetime
from typing import Dict


class CostTracker:
    """Track API costs for different models"""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-3.5-turbo": {
            "input": 0.0005,
            "output": 0.0015
        },
        "gpt-4": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-turbo": {
            "input": 0.01,
            "output": 0.03
        }
    }

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.cost_file = os.path.join(log_dir, "cost_tracking.json")
        self.session_costs = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "by_model": {},
            "by_complexity": {
                "low": {"cost": 0.0, "calls": 0},
                "moderate": {"cost": 0.0, "calls": 0},
                "high": {"cost": 0.0, "calls": 0}
            },
            "start_time": datetime.now().isoformat()
        }

    def add_api_call(self, model: str, input_tokens: int, output_tokens: int,
                     complexity: str = None):
        """Record an API call and calculate cost"""
        if model not in self.PRICING:
            print(
                f"Warning: Unknown model {model}, using gpt-3.5-turbo pricing")
            model = "gpt-3.5-turbo"

        # Calculate cost
        input_cost = (input_tokens / 1000) * self.PRICING[model]["input"]
        output_cost = (output_tokens / 1000) * self.PRICING[model]["output"]
        total_cost = input_cost + output_cost

        # Update totals
        self.session_costs["total_input_tokens"] += input_tokens
        self.session_costs["total_output_tokens"] += output_tokens
        self.session_costs["total_cost"] += total_cost

        # Update by model
        if model not in self.session_costs["by_model"]:
            self.session_costs["by_model"][model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "calls": 0
            }

        self.session_costs["by_model"][model]["input_tokens"] += input_tokens
        self.session_costs["by_model"][model]["output_tokens"] += output_tokens
        self.session_costs["by_model"][model]["cost"] += total_cost
        self.session_costs["by_model"][model]["calls"] += 1

        # Update by complexity
        if complexity and complexity in self.session_costs["by_complexity"]:
            self.session_costs["by_complexity"][complexity]["cost"] += total_cost
            self.session_costs["by_complexity"][complexity]["calls"] += 1

        # Save to file
        self.save()

        return total_cost

    def save(self):
        """Save current costs to file"""
        self.session_costs["last_updated"] = datetime.now().isoformat()
        with open(self.cost_file, 'w') as f:
            json.dump(self.session_costs, f, indent=2)

    def get_summary(self) -> Dict:
        """Get cost summary"""
        return {
            "total_cost": f"${self.session_costs['total_cost']:.4f}",
            "total_tokens": self.session_costs["total_input_tokens"] +
                          self.session_costs["total_output_tokens"],
            "by_complexity": {
                k: f"${v['cost']:.4f} ({v['calls']} calls)"
                for k, v in self.session_costs["by_complexity"].items()
            }
        }
