import asyncio
from typing import AsyncGenerator, Dict, Any

from haystack import AsyncPipeline
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from asyncio import Queue
import uuid
import json
import csv
from pathlib import Path

LOG_FILE = Path("logs/rag_log.csv")
LOG_FILE.parent.mkdir(exist_ok=True)

def log_interaction(question: str, answer: str):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["question", "answer"])  # header
        writer.writerow([question, answer])

# Document Store
document_store = QdrantDocumentStore(
    path="corpus/qdrant_data",
    index="educational_bot",
    embedding_dim=768,
    return_embedding=True,
    wait_result_from_api=True,
)

# Template
prompt_template = """
Given only the following information, answer the question.
Ignore your own knowledge.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ query }}
"""

# Chunk Collector for SSE
class ChunkCollector:
    def __init__(self):
        self.queue = Queue()

    async def generator(self) -> AsyncGenerator[str, None]:
        yield f'event: metadata\ndata: {{"run_id": "{uuid.uuid4()}"}}\n\n'
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                yield 'event: end\n\n'
                break
            yield f'event: data\ndata: {json.dumps(chunk)}\n\n'

async def collect_chunk(queue: Queue, chunk: StreamingChunk):
    if chunk and chunk.content:
        await queue.put(chunk.content)

# The Async Streaming Pipeline
async def stream_pipeline(question: str, top_k: int = 5) -> AsyncGenerator[str, None]:
    # Components
    embedder = OllamaTextEmbedder(model="nomic-embed-text", url="http://172.17.0.1:11434")
    retriever = QdrantEmbeddingRetriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template, required_variables=["documents", "query"])
    generator = OllamaGenerator(model="zephyr", url="http://172.17.0.1:11434", timeout=300)

    pipeline = AsyncPipeline()
    pipeline.add_component("query_embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")

    collector = ChunkCollector()
    loop = asyncio.get_running_loop()

    full_answer = []  # to accumulate output

    async def async_cb(chunk):
        if chunk and chunk.content:
            full_answer.append(chunk.content)
            await collect_chunk(collector.queue, chunk)

    def sync_cb(chunk):
        future = asyncio.run_coroutine_threadsafe(async_cb(chunk), loop)
        try:
            future.result()
        except Exception as e:
            print(f"Streaming callback error: {e}")

    input_data: Dict[str, Any] = {
        "query_embedder": {"text": question},
        "retriever": {"top_k": top_k},
        "prompt_builder": {"query": question},
        "generator": {"streaming_callback": sync_cb},
    }

    async def runner():
        try:
            async for _ in pipeline.run_async_generator(input_data):
                pass
        finally:
            await collector.queue.put(None)

    task = asyncio.create_task(runner())

    try:
        async for chunk in collector.generator():
            yield chunk
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Write to CSV after streaming ends
        log_interaction(question, "".join(full_answer))
