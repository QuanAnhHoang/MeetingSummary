import logging
import openai
import asyncio
import tiktoken
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import LLM_MODEL, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH
from config.constants import ERROR_MESSAGES, MAX_RETRY_ATTEMPTS

logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, api_key: Optional[str] = None, model: str = LLM_MODEL):
        self.model = model
        if api_key:
            openai.api_key = api_key
        self.encoding = tiktoken.encoding_for_model(model)
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit while maintaining coherence."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        # Try to end at a sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > 0:
            truncated_text = truncated_text[:last_period + 1]
            
        return truncated_text

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True
    )
    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate completion using OpenAI's API with retry logic.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty for token selection
            presence_penalty: Presence penalty for token selection
            stop: Stop sequences
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        try:
            start_time = datetime.now()
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that generates clear and concise text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "model": self.model,
                "processing_time": processing_time,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated completion in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise

    async def generate_meeting_summary(
        self,
        meeting_transcript: str,
        meeting_metadata: Dict[str, Any],
        summary_type: str = "full",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate a meeting summary using the transcript and metadata.
        
        Args:
            meeting_transcript: Full meeting transcript
            meeting_metadata: Dictionary containing meeting metadata
            summary_type: Type of summary to generate
            language: Target language for the summary
            
        Returns:
            Dictionary containing the summary and related information
        """
        # Prepare meeting context
        context = self._prepare_meeting_context(meeting_metadata)
        
        # Construct prompt based on summary type
        prompt = self._construct_summary_prompt(
            transcript=meeting_transcript,
            context=context,
            summary_type=summary_type,
            language=language
        )
        
        # Generate summary
        summary_result = await self.generate_completion(
            prompt=prompt,
            max_tokens=int(MAX_SUMMARY_LENGTH * 1.2),  # Allow some buffer
            temperature=0.5,  # Lower temperature for more focused output
            frequency_penalty=0.3,  # Reduce repetition
            presence_penalty=0.2
        )
        
        # Post-process summary
        processed_summary = self._post_process_summary(
            summary_result["text"],
            summary_type=summary_type,
            language=language
        )
        
        return {
            "summary": processed_summary,
            "metadata": {
                "source_length": len(meeting_transcript),
                "summary_length": len(processed_summary),
                "compression_ratio": len(processed_summary) / len(meeting_transcript),
                **summary_result
            }
        }

    def _prepare_meeting_context(self, metadata: Dict[str, Any]) -> str:
        """Prepare meeting context string from metadata."""
        context_parts = [
            f"Meeting Title: {metadata.get('title', 'Untitled')}",
            f"Date: {metadata.get('date', 'Unknown')}",
            f"Duration: {metadata.get('duration', 'Unknown')}",
            f"Participants: {', '.join(metadata.get('participants', []))}",
            f"Platform: {metadata.get('platform', 'Unknown')}"
        ]
        
        if agenda := metadata.get('agenda'):
            context_parts.append(f"Agenda: {agenda}")
            
        return "\n".join(context_parts)

    def _construct_summary_prompt(
        self,
        transcript: str,
        context: str,
        summary_type: str,
        language: str
    ) -> str:
        """Construct appropriate prompt based on summary type and language."""
        base_prompt = f"""
Context:
{context}

Transcript:
{transcript}

Please generate a {summary_type} summary of the meeting in {language}, including:
1. Key points discussed
2. Important decisions made
3. Action items and their assignees
4. Next steps

The summary should be:
- Clear and concise
- Well-structured
- Professional in tone
- Between {MIN_SUMMARY_LENGTH} and {MAX_SUMMARY_LENGTH} characters
"""
        
        if summary_type == "executive":
            base_prompt += "\nFocus on high-level strategic points and key decisions."
        elif summary_type == "action_items":
            base_prompt += "\nFocus on specific tasks, responsibilities, and deadlines."
        elif summary_type == "decisions":
            base_prompt += "\nFocus on key decisions made and their rationale."
            
        return base_prompt

    def _post_process_summary(
        self,
        summary: str,
        summary_type: str,
        language: str
    ) -> str:
        """Clean and format the generated summary."""
        # Remove any excess whitespace
        summary = " ".join(summary.split())
        
        # Ensure summary length constraints
        if len(summary) < MIN_SUMMARY_LENGTH:
            logger.warning("Generated summary is too short")
        elif len(summary) > MAX_SUMMARY_LENGTH:
            summary = self.truncate_to_token_limit(
                summary,
                self.count_tokens(summary) * MAX_SUMMARY_LENGTH // len(summary)
            )
        
        # Add section headers if missing
        if summary_type == "full" and "Key Points" not in summary:
            sections = ["Key Points:", "Decisions:", "Action Items:", "Next Steps:"]
            summary_parts = summary.split("\n\n")
            if len(summary_parts) >= len(sections):
                summary = "\n\n".join(f"{section}\n{part.strip()}"
                                    for section, part in zip(sections, summary_parts))
        
        return summary

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of meeting content."""
        prompt = f"""
Analyze the sentiment of the following text, considering:
1. Overall tone (positive/negative/neutral)
2. Key emotional indicators
3. Level of engagement
4. Areas of concern or controversy

Text:
{text}

Provide a detailed analysis with specific examples.
"""
        
        result = await self.generate_completion(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            "analysis": result["text"],
            "metadata": {k: v for k, v in result.items() if k != "text"}
        }

    async def extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Extract main topics discussed in the meeting."""
        prompt = f"""
Extract the {max_topics} most important topics discussed in the following text.
For each topic, provide a brief (1-2 words) label.

Text:
{text}

Format: Return only the topic labels, one per line.
"""
        
        result = await self.generate_completion(
            prompt=prompt,
            max_tokens=100,
            temperature=0.3
        )
        
        topics = [topic.strip() for topic in result["text"].split("\n")
                 if topic.strip()][:max_topics]
        
        return topics