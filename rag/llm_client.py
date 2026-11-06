"""
Provider-Independent LLM Client Interface for Football RAG
==========================================================
Supports Google Gemini API, OpenAI API, and a graceful
offline/disabled fallback when no API key is configured.
"""

import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_ANALYST = """You are an expert, objective Football Analyst and AI Match Evaluation Assistant.
You provide evidence-grounded tactical analysis based STRICTLY on retrieved factual records and verified machine learning predictions.

CRITICAL RULES:
1. SEPARATION OF CONCERNS:
   - Always clearly distinguish:
     (a) MATHEMATICAL MODEL PREDICTIONS: Exact probabilities output by the ML models.
     (b) VERIFIED HISTORICAL FACTS: Real match scores, dates, and statistics from retrieved evidence.
     (c) TACTICAL INTERPRETATION: Logical sports analysis synthesizing the evidence.
2. NEVER OVERRIDE PROBABILITIES:
   - Never alter, recalculate, or replace the machine learning model predictions or output probabilities.
3. CAUSATION vs CORRELATION:
   - Do not claim that historical head-to-head records or past form cause future match outcomes. Sports outcomes have high stochastic variance.
4. NO FABRICATION:
   - Do NOT invent player names, expected goals (xG), injuries, or tactical statistics not present in the provided evidence.
   - If retrieved evidence is insufficient to answer a specific detail, state the limitation clearly.
5. SOURCE ATTRIBUTION:
   - Reference the numbered sources (e.g., [1], [2]) whenever citing historical match outcomes or records.
"""

class BaseLLMClient(ABC):
    @property
    @abstractmethod
    def provider_name(self):
        pass

    @property
    @abstractmethod
    def is_available(self):
        pass

    @abstractmethod
    def generate_response(self, user_prompt, evidence_text="", system_prompt=SYSTEM_PROMPT_ANALYST):
        pass

class GeminiLLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model="gemini-3.8-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def provider_name(self):
        return f"Google Gemini ({self.model})"

    @property
    def is_available(self):
        return bool(self._client and self.api_key)

    def generate_response(self, user_prompt, evidence_text="", system_prompt=SYSTEM_PROMPT_ANALYST):
        if not self.is_available:
            return None
        nl = chr(10)
        prompt_content = f"{system_prompt}{nl}{nl}### RETRIEVED FACTUAL EVIDENCE:{nl}{evidence_text}{nl}{nl}### USER INQUIRY / MATCH ANALYSIS REQUEST:{nl}{user_prompt}{nl}"
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt_content
            )
            return response.text
        except Exception as e:
            return f"[Error calling Gemini API: {str(e)}]"

class OpenAILLMClient(BaseLLMClient):
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def provider_name(self):
        return f"OpenAI ({self.model})"

    @property
    def is_available(self):
        return bool(self._client and self.api_key)

    def generate_response(self, user_prompt, evidence_text="", system_prompt=SYSTEM_PROMPT_ANALYST):
        if not self.is_available:
            return None
        nl = chr(10)
        try:
            user_msg = f"EVIDENCE:{nl}{evidence_text}{nl}{nl}REQUEST:{nl}{user_prompt}"
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[Error calling OpenAI API: {str(e)}]"

class DisabledLLMClient(BaseLLMClient):
    @property
    def provider_name(self):
        return "Evidence-Only Mode (No API Key Configured)"

    @property
    def is_available(self):
        return False

    def generate_response(self, user_prompt, evidence_text="", system_prompt=SYSTEM_PROMPT_ANALYST):
        nl = chr(10)
        header = "ℹ️ **LLM Generation is currently disabled** (No API Key detected)."
        body = "The application operates in **Evidence-Only Mode**, retrieving verified historical facts from the ChromaDB vector database without hallucinating."
        evidence_header = "### 📋 Retrieved Factual Evidence:"
        tip = "💡 *To enable interactive LLM narrative generation, enter your GEMINI_API_KEY in the sidebar or in a .env file.*"
        return f"{header}{nl}{nl}{body}{nl}{nl}{evidence_header}{nl}{evidence_text}{nl}{nl}---{nl}{tip}"

def get_llm_client(custom_api_key=None, provider_preference=None):
    gemini_key = custom_api_key if custom_api_key else os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if provider_preference == "openai" and openai_key:
        client = OpenAILLMClient(api_key=openai_key)
        if client.is_available:
            return client
    if gemini_key:
        client = GeminiLLMClient(api_key=gemini_key)
        if client.is_available:
            return client
    if openai_key:
        client = OpenAILLMClient(api_key=openai_key)
        if client.is_available:
            return client
    return DisabledLLMClient()
