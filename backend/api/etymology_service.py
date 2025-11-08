"""
Etymology service that fetches word evolution data from:
1. External APIs (Wiktionary)
2. LLM providers (OpenAI, DeepSeek, Gemini, Claude)
3. Fallback to static CSV files
"""
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from api.config import settings

logger = logging.getLogger(__name__)


class EtymologyService:
    """Service for fetching word etymology and evolution data."""
    
    def __init__(self):
        self.llm_client = None
        self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on available API keys."""
        provider = settings.active_llm_provider
        
        if not provider:
            logger.warning("No LLM API key found. LLM features will be disabled.")
            return
        
        try:
            if provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=settings.openai_api_key)
                self.llm_model = settings.openai_model
                logger.info(f"Initialized OpenAI client with model {self.llm_model}")
            
            elif provider == "deepseek":
                from openai import OpenAI  # DeepSeek uses OpenAI-compatible API
                self.llm_client = OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                self.llm_model = settings.deepseek_model
                logger.info(f"Initialized DeepSeek client with model {self.llm_model}")
            
            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self.llm_client = genai.GenerativeModel(settings.gemini_model)
                self.llm_model = settings.gemini_model
                logger.info(f"Initialized Gemini client with model {self.llm_model}")
            
            elif provider == "anthropic":
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=settings.anthropic_api_key)
                self.llm_model = settings.anthropic_model
                logger.info(f"Initialized Anthropic client with model {self.llm_model}")
        
        except Exception as e:
            logger.error(f"Failed to initialize {provider} client: {e}")
            self.llm_client = None
    
    async def fetch_wiktionary_data(self, word: str) -> Optional[Dict[str, Any]]:
        """Fetch etymology data from Wiktionary API."""
        if not settings.use_external_apis:
            return None
        
        try:
            # Wiktionary REST API endpoint
            url = f"{settings.wiktionary_api_base}/page/definition/{word}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched Wiktionary data for '{word}'")
                return data
            else:
                logger.warning(f"Wiktionary API returned status {response.status_code} for '{word}'")
                return None
        
        except Exception as e:
            logger.error(f"Error fetching Wiktionary data for '{word}': {e}")
            return None
    
    def generate_era_contexts_with_llm(
        self, 
        word: str, 
        eras: List[str],
        num_examples: int = 5
    ) -> Dict[str, List[str]]:
        """
        Use LLM to generate contextual usage examples for a word across different eras.
        
        Args:
            word: The word to analyze
            eras: List of time periods (e.g., ["1900s", "1950s", "2020s"])
            num_examples: Number of examples per era
        
        Returns:
            Dictionary mapping era to list of contextual examples
        """
        if not settings.use_llm_etymology or not self.llm_client:
            logger.warning("LLM etymology disabled or no client available")
            return {}
        
        prompt = f"""Analyze how the word "{word}" was used and understood across different time periods.

For each era listed below, provide {num_examples} distinct contextual examples or definitions that capture how people in that era would have understood and used this word. Focus on the semantic nuances, connotations, and cultural context of each period.

Eras: {", ".join(eras)}

Format your response as JSON:
{{
  "1900s": [
    "example or definition 1",
    "example or definition 2",
    ...
  ],
  "2020s": [
    "example or definition 1",
    ...
  ]
}}

Important:
- Each entry should be a complete phrase or short sentence showing meaning/usage
- Capture the semantic shift and cultural context
- Be historically accurate
- Focus on how the meaning or connotation changed

Respond ONLY with valid JSON, no preamble or markdown.
"""
        
        try:
            provider = settings.active_llm_provider
            
            if provider in ["openai", "deepseek"]:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a historical linguist and etymologist."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                content = response.choices[0].message.content
            
            elif provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                content = response.text
            
            elif provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.content[0].text
            
            # Parse JSON response
            # Remove markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            logger.info(f"Successfully generated {len(result)} era contexts for '{word}' using {provider}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating LLM contexts for '{word}': {e}")
            return {}
    
    def load_csv_fallback(self, word: str, era: str) -> List[str]:
        """Load examples from CSV files as fallback."""
        csv_path = settings.data_path / f"{era}_{word}.csv"
        
        if not csv_path.exists():
            logger.warning(f"CSV fallback file not found: {csv_path}")
            return []
        
        try:
            examples = []
            with csv_path.open("r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(line)
            
            logger.info(f"Loaded {len(examples)} examples from CSV: {csv_path}")
            return examples
        
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return []
    
    async def get_word_evolution(
        self,
        word: str,
        eras: List[str],
        num_examples: int = 5
    ) -> Dict[str, List[str]]:
        """
        Get word evolution data across eras.
        
        Priority:
        1. Try LLM generation if enabled
        2. Try Wiktionary API if enabled
        3. Fallback to CSV files if enabled
        
        Returns:
            Dictionary mapping era to list of contextual examples
        """
        result = {}
        
        # Try LLM first
        if settings.use_llm_etymology:
            llm_result = self.generate_era_contexts_with_llm(word, eras, num_examples)
            if llm_result:
                return llm_result
        
        # Try external APIs
        if settings.use_external_apis:
            wiktionary_data = await self.fetch_wiktionary_data(word)
            if wiktionary_data:
                # Parse Wiktionary data (you'd need to implement parsing logic)
                # For now, we'll skip to CSV fallback
                pass
        
        # Fallback to CSV
        if settings.fallback_to_csv:
            for era in eras:
                examples = self.load_csv_fallback(word, era)
                if examples:
                    result[era] = examples[:num_examples]
        
        return result


# Global service instance
etymology_service = EtymologyService()