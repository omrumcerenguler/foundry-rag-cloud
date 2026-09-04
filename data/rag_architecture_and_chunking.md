# RAG Architecture, Chunking, and Grounding Best Practices

## Embedding Model Selection

Azure OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors and offers a strong balance between retrieval quality, storage footprint, and cost for most enterprise knowledge-base scenarios. Larger embedding models can improve semantic separation for highly specialized or multilingual corpora, but the incremental retrieval quality gain must be weighed against increased storage size and higher per-request embedding cost. Consistency matters more than raw dimensionality: mixing embeddings from different model versions inside the same vector index silently degrades cosine similarity comparisons, since the vector spaces are not guaranteed to be compatible.

## Cosine Similarity as a Ranking Metric

Cosine similarity measures the angle between two vectors rather than their magnitude, which makes it well suited to text embeddings where absolute vector length carries little semantic meaning. A normalized dot product between a query vector and a candidate vector yields a similarity score bounded between minus one and one, with values closer to one indicating stronger semantic alignment. Ranking candidates by this score and applying a confidence threshold allows a retrieval system to reject weakly related content rather than forcing every query to return a fixed number of results regardless of relevance.

## Chunking Strategy and Overlap

Splitting documents into bounded chunks balances two competing pressures: chunks must be small enough to fit within a model's context window alongside multiple retrieved passages, yet large enough to preserve enough surrounding context for a coherent answer. A common approach uses a fixed character or token budget per chunk, such as one thousand characters, combined with a smaller overlap window, such one hundred characters, so that a sentence spanning a chunk boundary is not stripped of context in both fragments. Word-boundary-aware splitting, rather than naive fixed-offset slicing, avoids severing sentences mid-word and produces more readable retrieved passages for citation display.

## Deterministic Chunk Provenance

Each chunk should retain its source file name, its sequential chunk index within that source document, and its exact character offset from the start of the file. This provenance metadata allows a system to reconstruct exactly which portion of which document contributed to a given answer, which is essential both for user trust and for debugging retrieval quality issues. Short documents naturally produce a single chunk with index zero; only documents exceeding the configured chunk size produce multiple sequential indices such as zero, one, and two.

## Grounding and Citation Validation

A grounded generation system constrains the model to answer strictly from retrieved context and requires the response to reference the specific source identifiers it drew from. After generation, the system should scan the model's output for citation markers matching the retrieved source identifiers; if no valid citation is found, the safest behavior is to discard the answer and return a controlled fallback message rather than presenting an unsupported claim as fact. This citation-verification step is what separates a merely plausible-sounding answer from one that is actually traceable to a specific passage in the knowledge base.

## Context Budget Management

Because retrieved passages compete for a limited prompt budget, a fixed character ceiling should be enforced across all concatenated context before it is sent to the generation model. When the budget is exhausted partway through assembling context, the system should stop including further passages rather than silently truncating a passage mid-sentence, since a broken passage can mislead the model into citing an incomplete source. Enforcing this budget deterministically also protects against unbounded token costs when a query happens to match an unusually large number of chunks.

## Confidence Thresholds and Fallback Behavior

Setting a minimum similarity threshold, rather than always returning the top-k results, prevents a retrieval system from confidently answering a question it has no relevant context for. When the highest-ranked result falls below the configured threshold, returning a clear "insufficient information" response is preferable to forcing the model to speculate from weakly related passages. This threshold should be tunable per deployment, since the appropriate cutoff varies with embedding model, corpus domain, and the acceptable trade-off between recall and precision for a given use case.
