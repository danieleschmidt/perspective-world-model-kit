"""
Performance optimization utilities for sentiment analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from functools import lru_cache, partial
import time
from dataclasses import dataclass
from ..utils.logging import get_logger
from .caching import get_sentiment_cache, get_async_sentiment_cache
from .validation import validate_text_input, validate_sentiment_scores

logger = get_logger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""
    batch_size: int = 32
    max_workers: int = 4
    use_gpu: bool = True
    enable_caching: bool = True
    timeout_seconds: float = 30.0


class BatchSentimentProcessor:
    """
    High-performance batch processing for sentiment analysis.
    """
    
    def __init__(
        self,
        sentiment_analyzer,
        config: Optional[BatchProcessingConfig] = None
    ):
        self.analyzer = sentiment_analyzer
        self.config = config or BatchProcessingConfig()
        
        # Setup processing resources
        self.device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Move model to device
        if hasattr(self.analyzer, 'transformer'):
            self.analyzer.transformer.to(self.device)
            self.analyzer.sentiment_head.to(self.device)
            
        # Cache for performance
        self.cache = get_sentiment_cache() if self.config.enable_caching else None
        
        logger.info(f"Batch processor initialized on {self.device} with {self.config.max_workers} workers")
        
    def process_batch_sync(
        self,
        texts: List[str],
        return_raw_logits: bool = False
    ) -> List[Dict[str, float]]:
        """
        Process a batch of texts synchronously with optimized batching.
        
        Args:
            texts: List of texts to analyze
            return_raw_logits: Whether to return raw logits instead of probabilities
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
            
        # Validate and filter texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            try:
                validated_text = validate_text_input(text)
                valid_texts.append(validated_text)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Invalid text at index {i}: {e}")
                
        if not valid_texts:
            return [{"negative": 0.33, "neutral": 0.34, "positive": 0.33}] * len(texts)
            
        # Check cache first
        cached_results = []
        texts_to_process = []
        cache_indices = []
        
        if self.cache:
            for i, text in enumerate(valid_texts):
                cached_result = self.cache.get_text_sentiment(text, self.analyzer.model_name)
                if cached_result:
                    cached_results.append((i, cached_result))
                else:
                    texts_to_process.append(text)
                    cache_indices.append(i)
        else:
            texts_to_process = valid_texts
            cache_indices = list(range(len(valid_texts)))
            
        # Process uncached texts in batches
        processed_results = []
        if texts_to_process:
            for batch_start in range(0, len(texts_to_process), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(texts_to_process))
                batch_texts = texts_to_process[batch_start:batch_end]
                
                batch_results = self._process_batch_internal(batch_texts, return_raw_logits)
                processed_results.extend(batch_results)
                
                # Cache results
                if self.cache:
                    for text, result in zip(batch_texts, batch_results):
                        self.cache.cache_text_sentiment(text, self.analyzer.model_name, result)
                        
        # Combine cached and processed results
        all_results = [None] * len(valid_texts)
        
        # Place cached results
        for i, result in cached_results:
            all_results[i] = result
            
        # Place processed results
        for i, result in zip(cache_indices, processed_results):
            all_results[i] = result
            
        # Map back to original indices (handle invalid texts)
        final_results = []
        valid_result_idx = 0
        
        for i in range(len(texts)):
            if i in valid_indices:
                final_results.append(all_results[valid_result_idx])
                valid_result_idx += 1
            else:
                # Default sentiment for invalid texts
                final_results.append({"negative": 0.33, "neutral": 0.34, "positive": 0.33})
                
        return final_results
        
    def _process_batch_internal(
        self,
        texts: List[str],
        return_raw_logits: bool = False
    ) -> List[Dict[str, float]]:
        """Internal batch processing with optimized tokenization and inference."""
        try:
            # Tokenize all texts at once
            tokenizer_outputs = self.analyzer.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            input_ids = tokenizer_outputs["input_ids"].to(self.device)
            attention_mask = tokenizer_outputs["attention_mask"].to(self.device)
            
            # Batch inference
            with torch.no_grad():
                logits = self.analyzer.forward(input_ids, attention_mask)
                
                if return_raw_logits:
                    return logits.cpu().numpy().tolist()
                    
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                
            # Convert to sentiment dictionaries
            sentiment_labels = ["negative", "neutral", "positive"]
            results = []
            
            for prob_row in probabilities:
                sentiment_dict = {
                    label: float(prob)
                    for label, prob in zip(sentiment_labels, prob_row)
                }
                validate_sentiment_scores(sentiment_dict)
                results.append(sentiment_dict)
                
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Return default sentiments
            default_sentiment = {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            return [default_sentiment] * len(texts)
            
    async def process_batch_async(
        self,
        texts: List[str],
        return_raw_logits: bool = False
    ) -> List[Dict[str, float]]:
        """
        Process batch asynchronously for non-blocking operation.
        """
        loop = asyncio.get_event_loop()
        
        # Use async cache if available
        async_cache = get_async_sentiment_cache() if self.config.enable_caching else None
        
        if async_cache:
            # Check cache asynchronously
            cached_results = await async_cache.batch_get_sentiments(texts, self.analyzer.model_name)
            
            # Identify texts that need processing
            texts_to_process = []
            for i, (text, cached_result) in enumerate(zip(texts, cached_results)):
                if cached_result is None:
                    texts_to_process.append((i, text))
                    
            # Process uncached texts
            if texts_to_process:
                indices, uncached_texts = zip(*texts_to_process)
                processed_results = await loop.run_in_executor(
                    self.executor,
                    self.process_batch_sync,
                    list(uncached_texts),
                    return_raw_logits
                )
                
                # Cache new results
                cache_data = list(zip(uncached_texts, processed_results))
                await async_cache.batch_cache_sentiments(cache_data, self.analyzer.model_name)
                
                # Merge results
                final_results = cached_results.copy()
                for idx, result in zip(indices, processed_results):
                    final_results[idx] = result
                    
                return final_results
            else:
                return cached_results
        else:
            # No caching, process directly
            return await loop.run_in_executor(
                self.executor,
                self.process_batch_sync,
                texts,
                return_raw_logits
            )
            
    def process_stream(
        self,
        text_stream: Any,
        callback: Callable[[str, Dict[str, float]], None],
        buffer_size: int = 100
    ) -> None:
        """
        Process streaming text data with buffering for efficiency.
        
        Args:
            text_stream: Iterator or stream of texts
            callback: Function called for each result
            buffer_size: Number of texts to buffer before processing
        """
        buffer = []
        
        try:
            for text in text_stream:
                buffer.append(text)
                
                if len(buffer) >= buffer_size:
                    results = self.process_batch_sync(buffer)
                    
                    for text, result in zip(buffer, results):
                        callback(text, result)
                        
                    buffer.clear()
                    
            # Process remaining texts in buffer
            if buffer:
                results = self.process_batch_sync(buffer)
                for text, result in zip(buffer, results):
                    callback(text, result)
                    
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = self.cache.get_comprehensive_stats() if self.cache else {}
        
        return {
            "device": str(self.device),
            "batch_size": self.config.batch_size,
            "max_workers": self.config.max_workers,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "cache_stats": cache_stats
        }


class ParallelSentimentProcessor:
    """
    Multi-process sentiment analysis for CPU-intensive workloads.
    """
    
    def __init__(
        self,
        num_processes: Optional[int] = None,
        chunk_size: int = 100
    ):
        self.num_processes = num_processes or multiprocessing.cpu_count()
        self.chunk_size = chunk_size
        
        logger.info(f"Parallel processor initialized with {self.num_processes} processes")
        
    def process_large_dataset(
        self,
        texts: List[str],
        analyzer_factory: Callable,
        timeout: Optional[float] = None
    ) -> List[Dict[str, float]]:
        """
        Process large dataset using multiple processes.
        
        Args:
            texts: List of texts to process
            analyzer_factory: Function that creates analyzer instances
            timeout: Processing timeout in seconds
            
        Returns:
            List of sentiment analysis results
        """
        if not texts:
            return []
            
        # Split texts into chunks
        chunks = [
            texts[i:i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, analyzer_factory): chunk
                for chunk in chunks
            }
            
            results = []
            completed_chunks = 0
            
            for future in as_completed(future_to_chunk, timeout=timeout):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    completed_chunks += 1
                    
                    if completed_chunks % 10 == 0:
                        logger.info(f"Completed {completed_chunks}/{len(chunks)} chunks")
                        
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Chunk processing failed: {e}")
                    
                    # Add default results for failed chunk
                    default_sentiment = {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
                    results.extend([default_sentiment] * len(chunk))
                    
        return results
        
    @staticmethod
    def _process_chunk(texts: List[str], analyzer_factory: Callable) -> List[Dict[str, float]]:
        """Process a chunk of texts in separate process."""
        try:
            analyzer = analyzer_factory()
            results = []
            
            for text in texts:
                try:
                    result = analyzer.analyze_text(text)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Text processing failed: {e}")
                    results.append({"negative": 0.33, "neutral": 0.34, "positive": 0.33})
                    
            return results
            
        except Exception as e:
            logger.error(f"Chunk processor initialization failed: {e}")
            default_sentiment = {"negative": 0.33, "neutral": 0.34, "positive": 0.33}
            return [default_sentiment] * len(texts)


@lru_cache(maxsize=1000)
def cached_tokenize(text: str, model_name: str) -> Tuple[List[int], List[int]]:
    """Cache tokenization results for frequently used texts."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    outputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    return (
        outputs["input_ids"].squeeze().tolist(),
        outputs["attention_mask"].squeeze().tolist()
    )


class OptimizedModelWrapper:
    """
    Wrapper for sentiment models with performance optimizations.
    """
    
    def __init__(self, model, enable_jit: bool = True, enable_half_precision: bool = False):
        self.model = model
        self.device = next(model.parameters()).device
        
        # JIT compilation for faster inference
        if enable_jit and hasattr(torch.jit, 'script'):
            try:
                self.model = torch.jit.script(model)
                logger.info("JIT compilation enabled")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}")
                
        # Half precision for GPU memory efficiency
        if enable_half_precision and self.device.type == 'cuda':
            self.model = self.model.half()
            logger.info("Half precision enabled")
            
        # Warmup
        self._warmup()
        
    def _warmup(self) -> None:
        """Warmup model with dummy input."""
        try:
            dummy_input = torch.randint(0, 1000, (1, 128)).to(self.device)
            dummy_mask = torch.ones_like(dummy_input).to(self.device)
            
            with torch.no_grad():
                self.model(dummy_input, dummy_mask)
                
            logger.debug("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass."""
        return self.model(input_ids, attention_mask)