import os
import logging
from typing import Optional, BinaryIO
import whisper
import torch
from pathlib import Path
from datetime import datetime
from config.settings import SPEECH_TO_TEXT_MODEL, TEMP_PATH
from config.constants import SUPPORTED_AUDIO_FORMATS, MAX_RETRY_ATTEMPTS

logger = logging.getLogger(__name__)

class SpeechToTextConverter:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_formats = SUPPORTED_AUDIO_FORMATS
        self.temp_dir = Path(TEMP_PATH)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            try:
                logger.info(f"Loading Whisper model: {SPEECH_TO_TEXT_MODEL}")
                self.model = whisper.load_model(SPEECH_TO_TEXT_MODEL, device=self.device)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {str(e)}")
                raise

    def validate_audio_file(self, file_path: str) -> bool:
        """Validate if the audio file format is supported and file exists."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported audio format. Supported formats: {', '.join(self.supported_formats)}")
            
        return True

    def preprocess_audio(self, file_path: str) -> str:
        """Preprocess audio file if needed (format conversion, noise reduction, etc.)."""
        # TODO: Implement audio preprocessing if needed
        return file_path

    async def transcribe_audio(
        self,
        file_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> dict:
        """
        Transcribe audio file to text using Whisper model.
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            task: Either 'transcribe' or 'translate'
            **kwargs: Additional parameters for Whisper model
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            self.validate_audio_file(file_path)
            self.load_model()
            
            processed_file = self.preprocess_audio(file_path)
            
            logger.info(f"Starting transcription for file: {file_path}")
            start_time = datetime.now()
            
            options = {
                "task": task,
                "language": language,
                **kwargs
            }
            
            result = self.model.transcribe(processed_file, **options)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            transcription_result = {
                "text": result["text"],
                "segments": result["segments"],
                "language": result["language"],
                "processing_time": processing_time,
                "model_name": SPEECH_TO_TEXT_MODEL,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    async def transcribe_stream(
        self,
        audio_stream: BinaryIO,
        chunk_size: int = 30,  # seconds
        **kwargs
    ) -> dict:
        """
        Transcribe audio stream in chunks.
        
        Args:
            audio_stream: Audio file stream
            chunk_size: Size of each chunk in seconds
            **kwargs: Additional parameters for transcribe_audio
            
        Returns:
            Dictionary containing transcription results
        """
        temp_file = self.temp_dir / f"stream_{datetime.now().timestamp()}.wav"
        
        try:
            # Save stream to temporary file
            with open(temp_file, 'wb') as f:
                f.write(audio_stream.read())
            
            # Transcribe the temporary file
            result = await self.transcribe_audio(str(temp_file), **kwargs)
            return result
            
        finally:
            # Cleanup temporary file
            if temp_file.exists():
                temp_file.unlink()

    def get_supported_languages(self) -> list:
        """Return list of supported languages."""
        self.load_model()
        return self.model.supported_languages()

    def estimate_processing_time(self, file_size: int) -> float:
        """
        Estimate processing time based on file size and available computing resources.
        
        Args:
            file_size: Size of audio file in bytes
            
        Returns:
            Estimated processing time in seconds
        """
        # Basic estimation formula - can be improved based on empirical data
        base_processing_factor = 0.5 if torch.cuda.is_available() else 2.0
        estimated_audio_duration = file_size / (16000 * 2)  # Assuming 16kHz, 16-bit audio
        return estimated_audio_duration * base_processing_factor

    async def batch_transcribe(
        self,
        file_paths: list[str],
        max_concurrent: int = 3,
        **kwargs
    ) -> list[dict]:
        """
        Transcribe multiple audio files concurrently.
        
        Args:
            file_paths: List of audio file paths
            max_concurrent: Maximum number of concurrent transcriptions
            **kwargs: Additional parameters for transcribe_audio
            
        Returns:
            List of transcription results
        """
        import asyncio
        from itertools import islice
        
        results = []
        
        for batch in self._batch_iterator(file_paths, max_concurrent):
            tasks = [self.transcribe_audio(file_path, **kwargs) for file_path in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results

    @staticmethod
    def _batch_iterator(iterable, batch_size):
        """Helper method to create batches from an iterable."""
        iterator = iter(iterable)
        while batch := list(islice(iterator, batch_size)):
            yield batch