from haystack import Pipeline
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    PPTXToDocument,
    JSONConverter
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pathlib import Path
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_indexing_pipeline(cfg: DictConfig):
    c = cfg.preprocess_corpus  # shorthand for clarity

    # Create document store from config
    document_store = QdrantDocumentStore(
        path=c.document_store.path,
        index=c.document_store.index,
        embedding_dim=c.document_store.embedding_dim,
        recreate_index=c.document_store.recreate_index,
        return_embedding=c.document_store.return_embedding,
        wait_result_from_api=c.document_store.wait_result_from_api,
    )

    # Define the pipeline
    pipeline = Pipeline()
    pipeline.add_component("router", FileTypeRouter(mime_types=c.file_types))

    # Converters
    pipeline.add_component("pdf_converter", PyPDFToDocument())
    pipeline.add_component("pptx_converter", PPTXToDocument())
    pipeline.add_component("txt_converter", TextFileToDocument())
    pipeline.add_component("json_converter", JSONConverter(
        jq_schema=c.converters.json.jq_schema,
        content_key=c.converters.json.content_key,
        extra_meta_fields=set(c.converters.json.extra_meta_fields)
    ))

    # Preprocessing
    pipeline.add_component("joiner", DocumentJoiner(join_mode=c.preprocessing.join_mode))
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(
        split_by=c.preprocessing.split_by,
        split_length=c.preprocessing.split_length
    ))

    # Embedder + Writer
    pipeline.add_component("embedder", OllamaDocumentEmbedder(
        model=c.embedding_model.name,
        url=c.embedding_model.url
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    # Connect components
    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx_converter.sources")
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect("router.application/json", "json_converter.sources")

    pipeline.connect("pdf_converter", "joiner")
    pipeline.connect("pptx_converter", "joiner")
    pipeline.connect("txt_converter", "joiner")
    pipeline.connect("json_converter", "joiner")

    pipeline.connect("joiner", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    # Load and run files
    corpus_dir = Path(c.corpus_path)
    files = list(corpus_dir.rglob("*.*"))
    pipeline.run({"router": {"sources": files}})

    # Return number of chunks
    print(f"Indexing complete. Documents stored: {len(document_store.filter_documents())}")

if __name__ == "__main__":
    run_indexing_pipeline()
