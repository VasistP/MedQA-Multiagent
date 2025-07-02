# src/agents/base_agent.py
import openai
from typing import Dict, List, Optional, Tuple
import tiktoken
import os


class BaseAgent:
    """Base class for all agents in the MDAgents system"""

    def __init__(self, role: str, instruction: str, model: str = "gpt-3.5-turbo",
                 few_shot_examples: List[Dict] = None, logger=None):
        self.role = role
        self.instruction = instruction
        self.model = model
        self.few_shot_examples = few_shot_examples or []
        self.logger = logger
        self.conversation_history = []

        # Initialize OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI()

        # Token counter
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))

    def create_messages(self, query: str) -> List[Dict]:
        """Create message list with system prompt, few-shot examples, and query"""
        messages = [{"role": "system", "content": self.instruction}]

        # Add few-shot examples
        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append(
                {"role": "assistant", "content": example["answer"]})

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    def chat(self, message: str, temperature: float = 0.7) -> Tuple[str, Dict]:
        """Send message to OpenAI API and get response"""
        messages = self.create_messages(message)

        # Count input tokens
        input_tokens = sum(self.count_tokens(m["content"]) for m in messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1500
            )

            # Extract response
            answer = response.choices[0].message.content
            output_tokens = self.count_tokens(answer)
            total_tokens = input_tokens + output_tokens

            # Log API call if logger is available
            if self.logger:
                complexity = getattr(self, 'current_complexity', None)
                self.logger.log_api_call(
                    total_tokens, self.model, input_tokens,
                    output_tokens, complexity
                )

            # Store in conversation history
            self.conversation_history.append({
                "role": self.role,
                "message": message,
                "response": answer,
                "tokens": total_tokens
            })

            return answer, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }

        except Exception as e:
            print(f"Error in API call: {e}")
            return "", {"error": str(e)}
