# src/utils/cost_tracker.py
import json
import os
from datetime import datetime
from typing import Dict


class CostTracker:
    """Track and monitor API costs for OpenAI model usage across the MedQA multi-agent system.

    This class provides comprehensive cost tracking capabilities for API calls made to different
    OpenAI models, with support for aggregating costs by model type and case complexity level.
    All costs are calculated based on token usage and stored persistently in a JSON file.

    Attributes:
        log_dir (str): Directory where cost tracking files are stored.
        cost_file (str): Full path to the cost tracking JSON file.
        session_costs (dict): Dictionary containing all cost tracking data for the current session.
            Structure:
                - total_input_tokens (int): Cumulative input tokens across all API calls
                - total_output_tokens (int): Cumulative output tokens across all API calls
                - total_cost (float): Total cost in USD for all API calls
                - by_model (dict): Per-model breakdown of usage and costs
                - by_complexity (dict): Per-complexity-level breakdown of usage and costs
                - start_time (str): ISO format timestamp of when tracking started
                - last_updated (str): ISO format timestamp of last update

    Pricing Information:
        PRICING (dict): Pricing per 1K tokens for different OpenAI models (as of 2024).
            Rates are in USD per 1,000 tokens:
            - gpt-3.5-turbo: $0.0005/1K input, $0.0015/1K output
            - gpt-4: $0.01/1K input, $0.03/1K output
            - gpt-4-turbo: $0.01/1K input, $0.03/1K output

    Example:
        >>> from src.utils.cost_tracker import CostTracker
        >>>
        >>> # Initialize cost tracker
        >>> tracker = CostTracker(log_dir="./logs")
        >>>
        >>> # Record an API call
        >>> cost = tracker.add_api_call(
        ...     model="gpt-4",
        ...     input_tokens=500,
        ...     output_tokens=200,
        ...     complexity="moderate"
        ... )
        >>> print(f"API call cost: ${cost:.4f}")
        API call cost: $0.0110
        >>>
        >>> # Get cost summary
        >>> summary = tracker.get_summary()
        >>> print(summary)
        {'total_cost': '$0.0110', 'total_tokens': 700, ...}

    Notes:
        - All costs are automatically saved to disk after each API call
        - If an unknown model is encountered, it defaults to gpt-3.5-turbo pricing
        - The tracker supports three complexity levels: 'low', 'moderate', 'high'
        - Token counts should be obtained from the OpenAI API response
    """

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
        """Initialize the CostTracker with a logging directory.

        Sets up cost tracking infrastructure including file paths and initializes
        the session cost tracking data structure. Creates a new tracking session
        with timestamp.

        Args:
            log_dir (str): Directory path where cost tracking files will be stored.
                          The directory should already exist; if not, file writing
                          operations will fail.

        Attributes Initialized:
            - log_dir: Stores the provided logging directory path
            - cost_file: Full path to 'cost_tracking.json' within log_dir
            - session_costs: Initialized data structure for tracking all costs

        Example:
            >>> tracker = CostTracker(log_dir="./experiment_logs")
            >>> print(tracker.cost_file)
            ./experiment_logs/cost_tracking.json
        """
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
        """Record an API call and calculate its cost.

        This method processes a single API call by:
        1. Calculating the cost based on input/output tokens and model pricing
        2. Updating cumulative statistics (total tokens, total cost)
        3. Updating per-model breakdown statistics
        4. Updating per-complexity breakdown statistics (if complexity provided)
        5. Persisting all data to disk

        Args:
            model (str): Name of the OpenAI model used for the API call.
                        Must be one of: 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'.
                        If an unknown model is provided, defaults to 'gpt-3.5-turbo'
                        pricing and prints a warning.
            input_tokens (int): Number of tokens in the input/prompt sent to the API.
                               Should be obtained from the API response's usage field.
            output_tokens (int): Number of tokens in the output/completion from the API.
                                Should be obtained from the API response's usage field.
            complexity (str, optional): Complexity level of the medical case being processed.
                                       Must be one of: 'low', 'moderate', 'high'.
                                       If None or invalid, complexity tracking is skipped.
                                       Defaults to None.

        Returns:
            float: The calculated cost in USD for this specific API call.
                  Cost = (input_tokens/1000 * input_rate) + (output_tokens/1000 * output_rate)

        Side Effects:
            - Updates self.session_costs with new totals and breakdowns
            - Writes updated cost data to disk via self.save()
            - Prints warning message if unknown model is encountered

        Example:
            >>> tracker = CostTracker(log_dir="./logs")
            >>> # Record a GPT-4 API call for a moderate complexity case
            >>> cost = tracker.add_api_call(
            ...     model="gpt-4",
            ...     input_tokens=1500,
            ...     output_tokens=300,
            ...     complexity="moderate"
            ... )
            >>> print(f"This call cost: ${cost:.4f}")
            This call cost: $0.0240
            >>>
            >>> # Record a call without complexity tracking
            >>> cost = tracker.add_api_call(
            ...     model="gpt-3.5-turbo",
            ...     input_tokens=800,
            ...     output_tokens=200
            ... )

        Notes:
            - Token counts should come from the OpenAI API response object:
              response.usage.prompt_tokens and response.usage.completion_tokens
            - All costs are calculated and stored in USD
            - Data is automatically persisted after each call
        """
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
        """Persist current cost tracking data to disk.

        Saves the complete session_costs dictionary to a JSON file, updating the
        'last_updated' timestamp before writing. The JSON file is formatted with
        2-space indentation for human readability.

        Side Effects:
            - Updates self.session_costs["last_updated"] with current timestamp
            - Writes/overwrites the cost tracking JSON file at self.cost_file
            - Creates the file if it doesn't exist; overwrites if it does

        Raises:
            OSError: If the log directory doesn't exist or lacks write permissions
            IOError: If file writing fails for any reason

        Example:
            >>> tracker = CostTracker(log_dir="./logs")
            >>> tracker.add_api_call("gpt-4", 1000, 500)
            >>> tracker.save()  # Explicitly save (though add_api_call auto-saves)

        Notes:
            - This method is called automatically by add_api_call()
            - Manual calling is only necessary if you modify session_costs directly
            - The JSON file can be read by external tools for cost analysis
        """
        self.session_costs["last_updated"] = datetime.now().isoformat()
        with open(self.cost_file, 'w') as f:
            json.dump(self.session_costs, f, indent=2)

    def get_summary(self) -> Dict:
        """Generate a formatted summary of cost tracking data.

        Creates a human-readable summary dictionary containing total costs,
        token usage, and breakdown by complexity level. All monetary values
        are formatted as USD strings with 4 decimal places.

        Returns:
            Dict: A dictionary containing:
                - total_cost (str): Total cost formatted as "$X.XXXX"
                - total_tokens (int): Sum of all input and output tokens
                - by_complexity (dict): Breakdown by complexity level, where each
                  entry is formatted as "$X.XXXX (N calls)" with:
                  - 'low': Cost and call count for low complexity cases
                  - 'moderate': Cost and call count for moderate complexity cases
                  - 'high': Cost and call count for high complexity cases

        Example:
            >>> tracker = CostTracker(log_dir="./logs")
            >>> tracker.add_api_call("gpt-4", 1000, 500, complexity="moderate")
            >>> tracker.add_api_call("gpt-4", 800, 300, complexity="low")
            >>> summary = tracker.get_summary()
            >>> print(summary)
            {
                'total_cost': '$0.0360',
                'total_tokens': 2600,
                'by_complexity': {
                    'low': '$0.0110 (1 calls)',
                    'moderate': '$0.0250 (1 calls)',
                    'high': '$0.0000 (0 calls)'
                }
            }
            >>> print(f"Total spent: {summary['total_cost']}")
            Total spent: $0.0360

        Notes:
            - This method does not modify any state; it's read-only
            - Complexity levels with 0 calls will still appear with $0.0000
            - Use this for end-of-session reporting or periodic cost checks
            - For more detailed breakdown, access self.session_costs directly
        """
        return {
            "total_cost": f"${self.session_costs['total_cost']:.4f}",
            "total_tokens": self.session_costs["total_input_tokens"] +
                          self.session_costs["total_output_tokens"],
            "by_complexity": {
                k: f"${v['cost']:.4f} ({v['calls']} calls)"
                for k, v in self.session_costs["by_complexity"].items()
            }
        }
