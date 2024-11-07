import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

from .speech_to_text import SpeechToTextConverter
from .llm import LLMProcessor
from models.meeting import MeetingContent, ContentType
from models.summary import (
    SummaryBase,
    SummaryCreate,
    ActionItem,
    Decision,
    SummaryKeyPoint,
    SummaryType
)
from config.settings import (
    STORAGE_PATH,
    MAX_SUMMARY_LENGTH,
    MIN_SUMMARY_LENGTH
)
from config.constants import ERROR_MESSAGES, MAX_RETRY_ATTEMPTS

logger = logging.getLogger(__name__)

class MeetingSummarizer:
    def __init__(
        self,
        speech_to_text: Optional[SpeechToTextConverter] = None,
        llm: Optional[LLMProcessor] = None
    ):
        self.speech_to_text = speech_to_text or SpeechToTextConverter()
        self.llm = llm or LLMProcessor()
        self.storage_path = Path(STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def process_meeting_content(
        self,
        content: List[MeetingContent],
        meeting_metadata: Dict[str, Any]
    ) -> str:
        """Process and combine different types of meeting content into a single transcript."""
        processed_contents = []

        for item in content:
            try:
                if item.type == ContentType.AUDIO:
                    transcript = await self._process_audio_content(item)
                elif item.type == ContentType.TRANSCRIPT:
                    transcript = await self._process_transcript_content(item)
                elif item.type == ContentType.CHAT:
                    transcript = await self._process_chat_content(item)
                else:
                    logger.warning(f"Unsupported content type: {item.type}")
                    continue

                processed_contents.append(transcript)
                item.processed = True

            except Exception as e:
                logger.error(f"Error processing content type {item.type}: {str(e)}")
                raise

        return self._combine_processed_content(processed_contents, meeting_metadata)

    async def _process_audio_content(self, content: MeetingContent) -> str:
        """Process audio content using speech-to-text conversion."""
        if content.content_url:
            # Download audio file if it's a URL
            audio_path = await self._download_audio(content.content_url)
        else:
            audio_path = content.content_text  # Assuming this is a local path

        transcription_result = await self.speech_to_text.transcribe_audio(
            file_path=audio_path,
            language=content.language
        )

        return transcription_result["text"]

    async def _process_transcript_content(self, content: MeetingContent) -> str:
        """Process transcript content."""
        if content.content_url:
            # Download transcript if it's a URL
            async with aiohttp.ClientSession() as session:
                async with session.get(content.content_url) as response:
                    return await response.text()
        return content.content_text

    async def _process_chat_content(self, content: MeetingContent) -> str:
        """Process chat messages into a coherent transcript."""
        chat_content = content.content_text
        if content.content_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(content.content_url) as response:
                    chat_content = await response.text()

        # Convert chat format to readable transcript
        chat_lines = chat_content.split('\n')
        formatted_chat = []

        for line in chat_lines:
            if line.strip():
                # Assume format: "[timestamp] username: message"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    formatted_chat.append(f"{parts[0].strip()}: {parts[1].strip()}")

        return "\n".join(formatted_chat)

    def _combine_processed_content(
        self,
        contents: List[str],
        metadata: Dict[str, Any]
    ) -> str:
        """Combine different processed contents into a single coherent transcript."""
        combined = "\n\n".join(contents)
        
        # Add metadata context
        context = [
            f"Meeting: {metadata.get('title', 'Untitled')}",
            f"Date: {metadata.get('date', 'Unknown')}",
            f"Participants: {', '.join(metadata.get('participants', []))}",
            "---",
            combined
        ]
        
        return "\n".join(context)

    async def generate_summary(
        self,
        meeting_content: List[MeetingContent],
        meeting_metadata: Dict[str, Any],
        summary_type: SummaryType = SummaryType.FULL,
        language: str = "en"
    ) -> SummaryCreate:
        """Generate a complete meeting summary."""
        try:
            # Process all content into a single transcript
            transcript = await self.process_meeting_content(
                meeting_content,
                meeting_metadata
            )

            # Generate initial summary using LLM
            summary_result = await self.llm.generate_meeting_summary(
                meeting_transcript=transcript,
                meeting_metadata=meeting_metadata,
                summary_type=summary_type.value,
                language=language
            )

            # Extract structured information
            key_points = await self._extract_key_points(summary_result["summary"])
            action_items = await self._extract_action_items(summary_result["summary"])
            decisions = await self._extract_decisions(summary_result["summary"])

            # Perform additional analysis
            sentiment = await self.llm.analyze_sentiment(transcript)
            topics = await self.llm.extract_topics(transcript)

            # Create summary object
            summary = SummaryCreate(
                meeting_id=meeting_metadata["meeting_id"],
                summary_type=summary_type,
                language=language,
                key_points=key_points,
                action_items=action_items,
                decisions=decisions,
                full_text=summary_result["summary"],
                generated_by="ai-summit",
                confidence_score=self._calculate_confidence_score(summary_result, sentiment)
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    async def _extract_key_points(self, summary: str) -> List[SummaryKeyPoint]:
        """Extract key points from the summary."""
        prompt = """
Extract key discussion points from the following summary.
For each point include:
1. Main topic
2. Brief description
3. Timestamp (if available)
4. Speaker (if available)

Format: Return as a list where each point has these components clearly labeled.

Summary:
{summary}
"""

        result = await self.llm.generate_completion(
            prompt=prompt.format(summary=summary),
            max_tokens=500,
            temperature=0.3
        )

        # Parse the response into SummaryKeyPoint objects
        key_points = []
        current_point = {}

        for line in result["text"].split("\n"):
            line = line.strip()
            if line:
                if line.startswith("Topic:"):
                    if current_point:
                        key_points.append(SummaryKeyPoint(**current_point))
                        current_point = {}
                    current_point["topic"] = line.replace("Topic:", "").strip()
                elif line.startswith("Description:"):
                    current_point["description"] = line.replace("Description:", "").strip()
                elif line.startswith("Timestamp:"):
                    try:
                        timestamp = line.replace("Timestamp:", "").strip()
                        current_point["timestamp"] = int(timestamp) if timestamp.isdigit() else None
                    except ValueError:
                        current_point["timestamp"] = None
                elif line.startswith("Speaker:"):
                    current_point["speaker"] = line.replace("Speaker:", "").strip()

        if current_point:
            key_points.append(SummaryKeyPoint(**current_point))

        return key_points

    async def _extract_action_items(self, summary: str) -> List[ActionItem]:
        """Extract action items from the summary."""
        prompt = """
Extract action items from the following summary.
For each action item include:
1. Description of the task
2. Assignee (if mentioned)
3. Due date (if mentioned)
4. Priority (if mentioned)

Format: Return as a list where each action item has these components clearly labeled.

Summary:
{summary}
"""

        result = await self.llm.generate_completion(
            prompt=prompt.format(summary=summary),
            max_tokens=500,
            temperature=0.3
        )

        # Parse the response into ActionItem objects
        action_items = []
        current_item = {}

        for line in result["text"].split("\n"):
            line = line.strip()
            if line:
                if line.startswith("Description:"):
                    if current_item:
                        action_items.append(ActionItem(**current_item))
                        current_item = {}
                    current_item["description"] = line.replace("Description:", "").strip()
                elif line.startswith("Assignee:"):
                    current_item["assignee"] = line.replace("Assignee:", "").strip()
                elif line.startswith("Due Date:"):
                    try:
                        due_date = datetime.strptime(
                            line.replace("Due Date:", "").strip(),
                            "%Y-%m-%d"
                        )
                        current_item["due_date"] = due_date
                    except ValueError:
                        current_item["due_date"] = None
                elif line.startswith("Priority:"):
                    current_item["priority"] = line.replace("Priority:", "").strip().lower()

        if current_item:
            action_items.append(ActionItem(**current_item))

        return action_items

    async def _extract_decisions(self, summary: str) -> List[Decision]:
        """Extract decisions from the summary."""
        prompt = """
Extract key decisions from the following summary.
For each decision include:
1. Topic
2. Description
3. Decision maker (if mentioned)
4. Impact areas
5. Stakeholders

Format: Return as a list where each decision has these components clearly labeled.

Summary:
{summary}
"""

        result = await self.llm.generate_completion(
            prompt=prompt.format(summary=summary),
            max_tokens=500,
            temperature=0.3
        )

        # Parse the response into Decision objects
        decisions = []
        current_decision = {}

        for line in result["text"].split("\n"):
            line = line.strip()
            if line:
                if line.startswith("Topic:"):
                    if current_decision:
                        decisions.append(Decision(**current_decision))
                        current_decision = {}
                    current_decision["topic"] = line.replace("Topic:", "").strip()
                elif line.startswith("Description:"):
                    current_decision["description"] = line.replace("Description:", "").strip()
                elif line.startswith("Made By:"):
                    current_decision["made_by"] = line.replace("Made By:", "").strip()
                elif line.startswith("Impact Areas:"):
                    areas = line.replace("Impact Areas:", "").strip()
                    current_decision["impact_areas"] = [a.strip() for a in areas.split(",")]
                elif line.startswith("Stakeholders:"):
                    stakeholders = line.replace("Stakeholders:", "").strip()
                    current_decision["stakeholders"] = [s.strip() for s in stakeholders.split(",")]

        if current_decision:
            decisions.append(Decision(**current_decision))

        return decisions

    def _calculate_confidence_score(
        self,
        summary_result: Dict[str, Any],
        sentiment_result: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on various metrics."""
        scores = []

        # Length-based score
        text_length = len(summary_result["summary"])
        length_score = min(1.0, max(0.0, 
            (text_length - MIN_SUMMARY_LENGTH) / (MAX_SUMMARY_LENGTH - MIN_SUMMARY_LENGTH)))
        scores.append(length_score)

        # Token usage score
        token_usage = summary_result.get("metadata", {}).get("total_tokens", 0)
        max_tokens = 4096  # Model's maximum context
        token_score = min(1.0, max(0.0, token_usage / max_tokens))
        scores.append(token_score)

        # Sentiment clarity score
        sentiment_text = sentiment_result.get("analysis", "").lower()
        sentiment_score = 0.8  # Default score
        if "unclear" in sentiment_text or "ambiguous" in sentiment_text:
            sentiment_score = 0.6
        scores.append(sentiment_score)

        # Calculate final score
        return sum(scores) / len(scores)

    async def _download_audio(self, url: str) -> str:
        """Download audio file from URL and save to storage."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    filename = url.split("/")[-1]
                    file_path = self.storage_path / filename
                    
                    with open(file_path, "wb") as f:
                        f.write(await response.read())
                    
                    return str(file_path)
                else:
                    raise ValueError(f"Failed to download audio file: {response.status}")