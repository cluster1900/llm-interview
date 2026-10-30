# Retrieval-Augmented Generation

[← Back to contents](../README.md)

## RAG-001. The model already stores a lot of public knowledge in its weights. Does an internal Q&A bot still need its own retrieval layer?

**Answer:** Yes. Parametric knowledge stops at the training cutoff, cannot see unpublished policies, tickets, or configs, and cannot replace one stale rule without retraining. A retrieval layer inserts currently citable evidence into the prompt at inference time, so answers can be updated, attributed, and sourced from context rather than from weights. Skip retrieval only for stable common-sense questions where staleness is acceptable; retrieve whenever the fact is private, the number must match a source, or a citation is required.

**Deep dive:** One turn looks like: user question → retriever pulls k passages from the enterprise corpus → a prompt builder trims to a token budget and labels sources → the generator answers on those passages. World knowledge in the weights still shapes wording and reasoning, so retrieval does not eliminate hallucinations; it only changes where facts are supposed to come from. Updating one policy via continued training usually means collecting samples, queuing a job, and running regressions—hours to days. Updating via retrieval means re-chunking and re-embedding that document, typically seconds to minutes. Boundary: if the corpus is empty, ACL filtering removes everything, or retrieval scores are unusable, refuse or say evidence is missing. Do not fall back to whatever the weights “remember”; that version may already be obsolete.

**Follow-ups:** Which internal questions can you explicitly mark as “no retrieval”? When a retrieved span disagrees with the model’s memory, how do you force the prompt to trust the span? If the retrieval layer is down, do you refuse, degrade to parametric answers, or serve a read-only cache, and why?

**Extended questions:**
- A policy changes every week. How do rollback paths differ if you refresh the index versus continuing to train the model?
- After you instruct the generator to use only numbers from retrieved spans, what extra checks still catch arithmetic that the model invents?
- Once retrieval is a routed on/off switch, which online metrics tell you a question class should stop retrieving?

**Reasoning:** List the hard constraints of internal Q&A: private, fresh, citable. Match them against three gaps in parametric knowledge—cutoff date, invisible private corpora, no surgical edit of one rule. Any gap justifies a retrieval layer. Compare the cost of updating one policy in the index versus in the weights, then keep “no evidence, no answer” so a stale parameter is never treated as the current rule.

## RAG-002. The model already writes acknowledgements in the company voice, but it still quotes last year’s prices as this quarter’s. Do you keep fine-tuning, or add retrieval?

**Answer:** Voice, fields, and stock phrases are stable behavior—fine-tune or harden a template. Prices, policies, and stock levels are mutable facts: retrieve them or call a system of record; do not store them in weights. Stop putting price lists into supervised data. Let retrieval or a tool emit numbers; keep fine-tuning for tone and format. You can stack both, but they own different layers, and retrieval wins conflicts.

**Deep dive:** Fine-tuning changes the conditional distribution of style, jargon, and task format. Retraining is days to weeks; a price list may move daily, and a fact in parameters is uncited and hard to delete cleanly. Retrieval or a price API changes the external state visible on each request and can carry an effective date. The opposite failure is also clear: if the model cannot fill your ticket JSON and keeps adding emoji, more price PDFs will not help—that is fine-tuning or constrained decoding. A workable split is small SFT/preference data for format and abstention wording, and a knowledge layer that retrieves; the prompt says numbers come from retrieval, not from remembered prices. Score style adherence, price accuracy, and citation rate separately. Domain fine-tuning is not a freshness strategy: it changes how you speak, not what the number is today.

**Follow-ups:** Can two hundred historic tickets stabilize voice; if retrieval has the right price and the model still rounds it to an integer, is that retrieval or generation; when you should query the trading system instead of a PDF.

**Extended questions:**
- Make a routing table: stale fact, bad format, missing jargon, safety abstention → retrieval, fine-tune, rules, or tool.
- Old prices leaked into the fine-tune set. How do you detect and scrub them after launch?
- When retrieval and parametric memory conflict, how do prompt and decoding constraints make retrieval win?

**Reasoning:** Classify the error: mutable fact versus stable behavior. Stale facts → retrieval or a tool; behavior → fine-tune or a template. Do not memorize price lists in supervised data. Accept on price accuracy and style adherence as separate scores.

## RAG-003. Employee-handbook Q&A goes live tomorrow. Which offline-index step and which online-ask step are each deal-breakers if missing?

**Answer:** Offline you must have repeatable ingest and parsing, structured chunking, a vector index (plus an optional lexical index), and a mapping from chunk IDs back to source text, section, and version. Online you must have authenticated retrieval, identity-aware filtering, context assembly with attributions, generation, and logs that record hit chunk IDs plus the model input and output. Without parse or index there is nothing to cite. Without filtering or logs you cannot prove there was no over-access and you cannot audit a bad answer.

**Deep dive:** Offline is a batch write path: connectors pull handbook files → a parser extracts body text and the heading tree → headers, footers, and duplicate TOCs are stripped → chunks are cut by clause or paragraph → an embedding model writes vectors in batches (for example 32–128 items, a few hundred to about a thousand dimensions) into an ANN index, while raw_text, doc_id, section, acl, and version go to a metadata table. Online is the read path: auth → optional query normalization → retrieve top-k (often 20–80 first) → optional rerank to a final 4–8 spans → pack the prompt to the generator window → call the LLM → return the answer and citations. The minimum launch bar is not “the model talks”. It is a few dozen real questions that each trace to a chunk ID, no unauthorized document ever entering the prompt, and index-build time plus query p95 inside the release budget. Monitor empty-retrieval rate, whether the newest document is searchable, and whether cited chunks still exist.

**Follow-ups:** If the metadata table and the vector index are one version apart, what does an online request see and how do you repair it? If v1 drops rerank and query rewriting, which step still cannot be dropped? The same clause has two effective versions—where on the read path must you pin the version?

**Extended questions:**
- Should parse failures and embedding failures retry separately or rerun the whole document, and how do you keep half-built chunks out of the index?
- What do you pay in recall and latency for pre-filtering ACL before retrieval versus post-filtering after retrieval?
- Which canary queries tell you that “the index is up but every answer is wrong” is a chunking bug versus a prompt-assembly bug?

**Reasoning:** Split the system into write path and read path. The write path must get evidence in and map it back to source by ID. The read path must fetch by identity, generate within budget, and remain auditable. Gate launch on “no ACL leak + chunk IDs can be replayed”. Treat rerank and rewrite as quality add-ons, not launch blockers.

## RAG-004. The embedding model accepts at most 512 tokens and the generator has an 8k window. You must both hit an exact refund clause and summarize a 20-page policy. How do you choose chunk length?

**Answer:** Do not share one length. Clause lookup wants short chunks so the key sentence occupies enough of the 512-token window—for example 128–256 tokens—with the clause ID in metadata. Summaries want larger semantic units: section-sized chunks near 400–512 tokens, or a short-chunk hit followed by fetching the enclosing section. 512 is a hard embedding cap: an over-long chunk is truncated at index time, so the second half never enters the vector.

**Deep dive:** When chunk length approaches 512, one relevant sentence plus neighboring noise pulls the vector off the clause, and exact-match recall drops. At 64–128 tokens, subjects, exceptions, and applicability often split, and generation sees half a sentence. A 20-page policy is typically thousands to about ten thousand tokens; an 8k generation window still has to hold the system prompt, dialogue, and citation markers, so it cannot take the whole book. Summaries should retrieve several section-sized chunks, not stuff the full text. Cut sections by headings first, then cut to the target length inside a section. Try 128 / 256 / 512 on the same question set and lock the length using clause recall@10 plus summary key-point coverage. Overlap is a secondary patch—about 10%–20% to cover boundary sentences—not a substitute for choosing length.

**Follow-ups:** Embedding truncation and generation truncation can both drop a tail—how do logs tell you which end lost it? If you retrieve a short chunk and then fetch the section, do you store two texts in the index? For mixed Chinese and English, why is token-based cutting more stable than character-based cutting?

**Extended questions:**
- The refund exception lives in the next section. After a short chunk hits the main clause, what fetch rule brings the exception into the generator context?
- About how many 512-token chunks fit in an 8k window once system prompt and citation markers are reserved?
- If you switch to an embedder with an 8k input, would you jump chunk size to 2k, and why or why not?

**Reasoning:** Write the two hard constraints: embeddings must not truncate away evidence, and the generator window must hold several chunks at once. Split by question type—short for exact match, wider for synthesis—and satisfy both with short-chunk recall plus section-level fetch instead of one magic default. Choose the length from split offline recall and key-point coverage.

## RAG-005. A runbook chunk only says “set the timeout to 30 seconds and restart the worker,” but users ask with a service name and change-ticket id. What do you add at index time so both lexical and dense retrieval can find that chunk?

**Answer:** Once the chunk leaves the full document, document-level referents such as the service name and ticket id are gone. Before writing the lexical index and the vector index, generate a short, standalone prefix that states the service, section, environment, and related change, then concatenate prefix plus original text and send that string through both BM25 and the embedder. Query terms for the service and ticket can then match the prefix’s wording and meaning; the action sentence does not have to repeat those entities.

**Deep dive:** When writing the prefix, put the chapter or the whole runbook in the prompt and ask for one or two sentences that locate the span and name what its pronouns refer to—for example “this span is from the payments-settlement online incident runbook, change ticket CHG-1842, discussing gateway timeouts.” Keep the prefix to a few dozen tokens so it does not drown the action. Chunks from the same document share a document prefix, so index-time prompt cache can reuse it; the main cost is one short generation per chunk, and query time does not call a model for prefixes. On an edit, regenerate prefixes only for affected chunks and update both indexes. Do not write a long abstract or BM25 will chase summary words. Do not store the service name only as a metadata filter: when users put the name inside the question, a filter field does not help lexical or vector match. If the document exceeds the generator window, cut by chapter first, then write prefixes inside each chapter.

**Follow-ups:** If you only prepend filename and heading, without a model-written prefix, how many of these chunks do you still recover? If the prefix goes into both BM25 and the vector, can chapter-level stopwords pull back the entire chapter? The runbook changes weekly—how do you avoid rewriting every prefix in the corpus?

**Extended questions:**
- How would you build de-contextualized query–chunk pairs to quantify recall before and after prefixes?
- Among policies, chat logs, and code comments, which class most deserves this index-time generation cost?
- If the prefix model is stronger than the retrieval embedder, when is that worth it, and how do you amortize the cost per document?

**Reasoning:** Start from the failure: query terms sit on document-level entities, while the chunk body is only actions and pronouns. Those entities must be written back into searchable text at index time. Compare metadata filters, human headings, and model-written prefixes: only text that enters both lexical and vector bodies can serve both retrievers, and a short prefix keeps the summary from taking over the chunk.

## RAG-006. You have ~40 product manuals of ~20 pages that barely change, and prompt-cache hit rates are high. Do you still retrieve, or stuff whole manuals into the window? What if each manual becomes a daily edition?

**Answer:** If the manuals for the current product fit the window with room for the reply, the text is almost static, and prefix cache amortizes prefill, stuffing one or a few whole manuals is simpler and better for cross-section comparison. A daily edition kills that cache every day and requires old pages to vanish immediately—then retrieve with a version filter. Per-user chapter cuts and paragraph-level citations also push you back to retrieval.

**Deep dive:** Do a capacity estimate; do not trust the sticker context length. At roughly 500–800 tokens per page, 20 pages × 40 manuals is about 400K–640K tokens, which still will not fit many long-context models, so you route by product and stuff 1–3 manuals, not all 40. Stuffing avoids chunk-boundary maintenance, keeps in-book comparison intact, and makes repeat requests almost incremental when the prefix is stable. It also underuses middle chapters, turns irrelevant chapters into noise, makes cold prefill expensive, and tends to cite at document granularity. Retrieval fetches on demand, cites at chunk granularity, and rebuilds only dirty documents. After a daily edition, the cache key must include the manual version, hit rate falls, and stuffing’s cost advantage disappears; retrieval just filters on version. Decide with your own set on accuracy, P95, unit cost, and unauthorized inclusion—not by comparing advertised window sizes. This is a traffic-routing decision, not a “how to run a fair bake-off” methods question.

**Follow-ups:** If you stuff exactly one manual, do you still retrieve inside chapters; what a username in the cached prefix does to hit rate; for a cross-manual comparison, whether stuffing 3 books or retrieving top-20 is stabler.

**Extended questions:**
- One manual grows from 20 to 200 pages with a nominal 128K window. How do you redo capacity and a position sweep?
- Full-document summary, clause-by-clause compare, and single-clause lookup: which way does each lean?
- How does security acceptance differ for “ACL-trim then stuff” versus retrieval that enforces access control?

**Reasoning:** First ask whether the tokens truly fit with output reserved, then whether update frequency still allows a cache. Fit and stable → stuff; fast updates, ACL, or span-level citations → retrieve. Run both on your load; do not assume either side is forever cheaper.

## RAG-007. Now that long-context models actually work, what yardsticks put full-document stuffing and RAG on a fair comparison?
**Answer:** On the same task set, compare answer correctness, evidence coverage, permission isolation, latency, and total cost; stratify by document length, update frequency, and multi-hop need. Do not compare only the model's per-call fee.
**Deep dive:** RAG has a missed-retrieval risk; full-text input has noise and long-context degradation risks. A retrieve-then-expand-context approach can be used.
**Follow-ups:** Which tasks suit dynamic routing between the two paths? You can choose by document length, how scattered the evidence is, update frequency, and the quality bar, but the routing rules themselves also need evaluation.
**Extended questions:**
- If the full-text approach is right and RAG is wrong, how do you distinguish missed retrieval, chunking damage, and generation error?
- How do you fold index maintenance, document-update, and permission-filtering costs into the total-cost comparison of the two approaches?
**Reasoning:** Use real workloads to find the cost-quality frontier; do not assume that full-text input or RAG is inherently better.

## RAG-008. A user says the timeout seconds in the answer are wrong. In what order do you open the pipeline, and what signal belongs to each stage?

**Answer:** First decide whether the evidence never entered the index, entered but was not retrieved, was retrieved but never reached the prompt, or was already correct in the prompt while the model still wrote the wrong number. Parsing and chunking: is the source number intact inside the chunk? Retrieval: did the gold chunk enter top-k? Rerank and assembly: did it survive truncation into the model input? Generation: if the input is right, does the output still drift? Do not swap in a larger generator until those earlier stages are cleared.

**Deep dive:** Keep one diagnostic case: the question, the gold chunk_id, and the number that must appear (for example the handbook says 30 seconds). Parse failure: the number lived in a table or two-column layout and the chunk is a broken sentence or the number is gone. Chunk failure: “30” and “seconds” split apart, or the exception landed in the next chunk. Retrieval failure: the gold chunk ranks beyond k; check at least k = 20 / 50 / 100. Rerank or assembly failure: the chunk ranked high in first-stage retrieval and was dropped when the token budget was applied. Generation failure: the model input already says “timeout 30 seconds” and the answer says 3 or 300. Each stage needs structured logs—chunk_ids, scores, truncation offset, hash of the model input—so one bad number maps to one layer. Replay that single trace before you re-embed the whole corpus.

**Follow-ups:** Recall metrics already pass but the number is still wrong—what generation-side fields do you assert on? If tables flatten into shuffled plaintext, which document shapes must the regression set add? The same number disagrees across two chunks—which layer must not silently pick one?

**Extended questions:**
- How do you replay retrieval only, without calling the generator, so diagnosis cost is not the same as generation spend?
- What layouts must parser and chunker unit tests cover at minimum: plain clauses, tables, headers and footers?
- If production keeps full traces for only 1% of requests, how do you estimate the chance that a rare truncation is never sampled?

**Reasoning:** The goal is to find the layer where the correct number left the source. Treat the pipeline as a one-way data flow and assert gold chunk ID plus target number at each hop: source integrity → retrieval coverage → prompt retention → generation consistency. Stop and fix at the first failed assertion so a chunking or truncation bug is not treated as a prompting bug.

## RAG-009. Contracts have nested clauses, FAQs are short Q-A pairs, and meeting notes barely have headings. How do you assign fixed-length cuts, semantic-boundary cuts, and heading-recursive cuts to the three corpora?

**Answer:** Cut FAQs on Q-A pairs, or with a fixed window, because the retrieval unit is already one pair. Cut contracts recursively on headings and clause numbers—chapter, then clause, then sentence—until the size cap is met, so a proviso is not split from its rule. Meeting notes have no stable tree: use fixed length on sentence boundaries, or split where adjacent-sentence embedding similarity drops. Do not invent a table of contents they do not have.

**Deep dive:** Fixed length is a sliding window of N tokens with optional overlap. Chunk sizes stay even, so index size and batched embedding time are predictable; table rows, numbered lists, and “exception to clause 7.2” get cut in half. Recursive cuts follow the parser’s heading tree: emit a node if it fits, otherwise split on child headings or paragraphs. Contracts and policies gain most when chunk borders match reading units. Semantic cuts embed adjacent sentences and split below a threshold calibrated on your own corpus; they help when notes change topic, but embedding calls grow with sentence count and indexing costs more than fixed windows. Mix them: recurse on headings you trust, then fixed-cut the leftover body. Do not accept a run by average chunk length alone—sample whether exceptions stay with their rule and whether FAQ questions were split from their answers.

**Follow-ups:** If the parser misreads clause titles, how does recursive cutting get worse, and what is the fallback? If the semantic threshold is too high or too low, what does the chunk-length histogram become? On a roughly 100-page contract, how far apart are the three methods in chunk count and index time?

**Extended questions:**
- The contract is a scan with layout but no real heading tree. Do you still insist on recursive cuts?
- Must the embedder used for semantic cuts be the same model used for online retrieval?
- After cutting, chunk-length variance is large. What does that do to batched embedding and ANN index build?

**Reasoning:** Start from whether the document has a trustworthy structure signal. A heading tree → recurse so chunks align with clauses. Ready-made Q-A pairs → cut on pairs and skip extra design. No structure → trade cost against topic integrity between fixed and semantic cuts. Accept the job with structural assertions such as “was the exception split,” not with the reputation of an algorithm.

## RAG-010. Clause-level retrieval is accurate, yet generation keeps missing definitions and exceptions in the same section — why split retrieval granularity from the context you feed the model, instead of just enlarging chunks?

**Answer:** Retrieval wants distinguishable short vectors; generation wants a complete discourse unit. Definitions and exceptions often sit two or three paragraphs away from the hit sentence. Once a single chunk is large enough to hold the whole section, the embedding averages adjacent clauses and matching gets worse. The split is: child chunks (about 100–250 tokens) only retrieve; on a hit, map to the parent (a section or page, about 800–1500 tokens) for the model. Enlarging chunks sacrifices retrievability and readability on the same knob; parent–child gives each objective its own window.

**Deep dive:** At ingest, cut parents on headings, then children inside each parent; child vectors go into the ANN with `parent_id` and character offsets. At query time: ANN on children → unique parents → concatenate in source order. Expand at most a neighboring sentence inside the same parent, not the whole chapter. Policies, manuals, and contracts benefit most because “For the purposes of this clause, X means…” and “except in the following cases” rarely share the hitting window. Edge cases: a parent that grows to a full chapter (thousands of tokens) brings noise and lost-in-the-middle back; many children mapping to one parent can collapse top-8 children into two parents—high surface recall, thin evidence. Parents must not be stitched across permission or version boundaries. Do not fold this into the same layer as sliding-window overlap: overlap fixes cuts; parent–child fixes two granularities.

**Follow-ups:** Do you cut parents by section or by page, and what do you do with untitled meeting notes? When top-8 children collapse to two parents, do you retrieve more or keep only the best child per parent? If child and parent ACLs disagree, which wins?

**Extended questions:**
- With 128-token children and 2k-token parents, should the reranker score the child or the expanded parent?
- If a parent also contains unrelated sibling clauses, how do you trim after expansion without cutting the exception sentence?
- How would you use the rate of “child hit but parent missing the definition” to show parent–child beats simply enlarging chunks?

**Reasoning:** Split the goal into retrieval distinctiveness versus generation completeness; those pull window length in opposite directions. Compare a single large embedding with “small-child retrieve + parent feed,” and watch noise as parents grow. Accept with child recall@k and whether the parent actually contains the definition or exception.

## RAG-011. Some people search “RFC-214 section 4.2,” others search “how do I refresh an expired token.” What does each of vector-only and keyword-only miss, and how do you merge the two lists into one?

**Answer:** Identifiers—RFC numbers, section numbers, error codes—belong on a lexical retriever such as BM25; if the rare literal is absent from the chunk, it almost never comes back. Paraphrased how-to questions belong on dense retrieval, which can match refresh / rotate / renew when the wording differs. Take top-n from each list and fuse by rank, not raw score (for example Reciprocal Rank Fusion), then truncate for rerank or generation. Tuning a single score threshold almost always helps one class and hurts the other.

**Deep dive:** BM25 uses term frequency and inverse document frequency, so it is sharp on rare tokens such as RFC-214, 4.2, or ECONNRESET, and weak on synonyms and mild typos. Dense retrieval can map “how do I refresh an expired token” onto a paragraph about rotating expired credentials, and can also promote another ticket that happens to contain 214, or a section-4 overview. RRF adds Σ 1/(κ + rank_i) per document, with κ in the tens (implementations often use about 60) so an extreme rank-1 does not dominate. The raw score scales differ: cosine is about -1 to 1, BM25 can reach tens, so a weighted sum needs per-channel normalization. Taking 20–50 per channel and keeping 20–80 after fusion is enough for many enterprise corpora. When a user embeds the identifier mid-sentence and a metadata extractor misses it, fusion can still keep the document via the lexical channel.

**Follow-ups:** If κ is much larger or smaller, which way do rare identifiers move in the fused ranking? If Chinese segmentation splits “token expired” badly, can the entire BM25 channel fail? When the two hit lists overlap heavily, is fusion still worth running?

**Extended questions:**
- If the sparse side moves from BM25 to model-estimated term weights, which literal failures does that repair?
- How would you report “identifier queries” versus “paraphrase queries” separately to prove fusion wins on both sides, not just on one blended score?
- If you fuse before versus after ACL filtering, what happens to empty-result rate and leak risk?

**Reasoning:** Use the two real query classes to show each channel’s blind spot: identifiers need literals, procedures need semantics. Keep both recall lists. Fuse on ranks so incomparable scores never meet, and prove both channels work with sliced metrics instead of one weighted total.

## RAG-012. After product lines grow from 12 to 80, retrieval starts mixing in other SKUs. Should the filter be an ANN pre-filter or a post-cut? Can ACL fields use the same pattern?

**Answer:** Relevance filters—product line, language, region, time—can be pre-filters or over-fetch-then-cut, depending on how selective they are and whether the index supports filtered ANN. ACL and tenant fields cannot be “retrieve then drop”: a sensitive chunk in the prompt is already a leak. Mixed SKUs mean the query is missing a scope predicate. Write metadata correctly at ingest and attach the current product scope on every search.

**Deep dive:** Pre-filtering applies a payload predicate during ANN so neighbors are drawn only from the subset. A tight predicate (`sku_id = current`, a few thousand chunks) is both fast and clean. If the engine really does global search plus a bitset, a very tight filter can wipe the candidate list and crush recall—over-fetch (ask for a couple of hundred to keep 20) or shard the index by product. Post-filtering is simple but spends distance computations on the wrong SKUs and can return fewer than k hits. Keep fields structured and stably enumerated; do not equality-filter free-text specs. Time is a range predicate. Authorization is a separate path: chunks carry allow-lists, the query injects the principal, the index enforces hits, caches are isolated per user. Eval on a distractor set of wrong SKUs: did the answer cite an out-of-scope document, and did recall@k collapse after the filter.

**Follow-ups:** The user never named the product, only session state has the SKU—where is the filter injected; do you relax and retry on a pre-filter zero; if one document belongs to several lines, is metadata multi-valued or do you split chunks.

**Extended questions:**
- When is a per-product shard better than one global index plus a filter?
- How do you combine the filter with the vector score so in-scope but off-topic chunks do not dominate?
- What exfiltration tests prove ACL is not implemented as retrieve-then-drop?

**Reasoning:** Split the purpose of the filter. Relevance can trade pre-filter against post-cut. Authorization must be enforced before retrieval. More SKUs make the predicate selective, so prefer shards or filtered ANN, and accept on the out-of-scope citation rate.

## RAG-013. A product query carries brand, size, and “not for children” at once, and a single vector often smears those three constraints into “same topic.” How does token-level late interaction score a description, and about how many vectors do you store for one million descriptions averaging 200 tokens?

**Answer:** Late interaction does not crush a description into one point; query and document keep a vector per token. At score time, each query token takes its maximum similarity over all document tokens (MaxSim), and those maxima are summed. Brand, size, and “child / not” can align to different positions in the description. Storage: 1e6 × 200 tokens is about 2×10^8 token vectors—roughly two orders of magnitude more than one vector per product.

**Deep dive:** Encoding still runs query and document separately, so document token vectors can be computed offline and interaction is delayed until scoring. With normalized q_i and d_j, s(q,d) = Σ_i max_j (q_i · d_j). If one query constraint has no near neighbor in the description, that term contributes near zero and the total drops; a single vector can still rank high because the brand was close. Illustrative space: a 128-d float16 token vector is 256 bytes, so 2×10^8 tokens are about 51 GB of payload before postings or clusters. One 768-d float16 vector per document is only about 1.5 GB. Production systems shrink this with residual compression and clustered pruning, but it stays far above single-vector indexes. A query is encoded once, then pruned MaxSim runs on the precomputed index—usable as first-stage retrieval or as a rerank over tens to hundreds of candidates. Very short descriptions have few tokens and weaker separation; long queries add more summed terms, so you must decide in advance whether to normalize by query length.

**Follow-ups:** Does MaxSim systematically favor longer descriptions, and should you divide by the number of query tokens? If token vectors are not normalized and you switch to L2 distance, how does the retrieval structure change? Against “single-vector recall + cross-encoder rerank,” what traffic volume makes this storage cost worthwhile?

**Extended questions:**
- If you drop function-word token vectors from an average 200-token description at index time, which query class do you hurt?
- After “not for children” splits into subwords, how do you stop MaxSim from aligning only to “children” and ignoring the negation?
- When a single product description changes, why is a local update harder on a token-level index than on a single-vector index?

**Reasoning:** A single vector cannot hold three hard constraints at once, so keep tokens and interact at score time. MaxSim explains how those constraints align separately; multiplying 1e6 × 200 turns the storage cost into a number you can trade against quality.

## RAG-014. The user wrote “a quote excluding tax”; the rewriter expanded it to “quotation process” and you only retrieve policy how-tos. How do you constrain query transformation without dropping negation and entities?

**Answer:** A rewrite exists to add retrieval language—synonyms, sub-questions, a hypothetical paragraph—not to promote the user’s intent into a topic. Keep negation, entities, numbers, and constraints, and always keep the raw question as one query. “Excluding tax” should become retrieval phrases like “pre-tax unit price / tax-exclusive quote,” not “quotation process.” Ship a rewrite gate: entities still present, polarity not flipped.

**Deep dive:** Different transforms do different jobs. Multi-query (raw plus two or three paraphrases, then fuse) is the safest default. Decomposition splits a multi-hop question into independently searchable parts. Step-back abstraction is dangerous on constrained questions like “excluding tax” because it strips the modifier. HyDE writes a hypothetical answer and embeds it; it helps conceptual questions that lack keywords, but it is weak on tax, part numbers, and negation that must match literally, so run it beside a lexical path. Drift happens when a generative rewrite is ungrounded and turns “not X” into “the process for X.” Constrain it: extract slots (product, tax-inclusive flag, currency) and rewrite structurally; check entity and negation coverage after the rewrite and fall back to the raw question on failure; keep a keyword search on the raw question in the fusion. Score context recall before versus after rewrite, plus a set built to detect dropped constraints—not how readable the rewrite is. The rewrite is usually one small-model call and can run beside intent classification.

**Follow-ups:** If HyDE’s hypothetical invents a number, does that pull the vector into the wrong cluster; how fusion stops one document from filling the top-k; whether a model number mentioned earlier in the dialogue should be written into the rewrite.

**Extended questions:**
- Write a rule: when a negation is present, forbid step-back and allow only synonym expansion.
- How do you automatically detect a polarity flip such as tax-inclusive versus tax-exclusive?
- If one decomposed subquery fails, do you still pass partial evidence to generation?

**Reasoning:** The objective is more retrievable language without changing constraints. Lock the slots that must survive, then allow paraphrases. Keep the raw question as one path. Accept the rewriter on constraint-loss rate, not fluency.

## RAG-015. Engineers ask for a “circuit-breaker threshold” and retrieve electrical-fuse documents; internal product codenames also get washed out in the vector space. Do you change query expansion, the glossary, or the embedding space, and how do you avoid pinning common words to one sense?

**Answer:** Keep the raw term and run hybrid search: the lexical channel holds the exact tokens for the jargon and the codename; the vector channel supplies paraphrases. Query expansion may only use a controlled glossary scoped to a workspace sense; a rewrite model must not generalize the term into “disconnect / fuse / protection.” If that is not enough, index short definition chunks, or continue-train the encoder on domain pairs. Do not globally rewrite the term to one sense—pinning it to microservice shedding makes electrical-safety questions fail systematically.

**Deep dive:** Failures split into polysemy (service breaker vs fuse vs trading halt) and rare codenames almost unseen in pretraining. Hybrid search helps the second immediately because BM25 does not need a semantic neighborhood. The first needs context: glossary entries carry a qualifier (“service governance / trading / electrical”), or a classifier picks a workspace before enabling that synonym set. A conservative vector-side fix is to retrieve against “term + definition + forbidden sense” mini-docs rather than changing the tokenizer. Only then fine-tune embeddings with domain contrastive pairs, and watch generic-query regression. Edge cases: automatic synonym expansion translates a codename into a common noun; spell-fix mutates a valid internal abbreviation. Slice the eval set into abbreviations, aliases, historical names, and jargon-free general questions.

**Follow-ups:** If workspace classification is wrong and the trading glossary is applied to an engineering query, how do you fall back to “keep the raw token only”? After domain fine-tuning, how much generic-FAQ recall loss is a no-ship? Who owns the glossary, and do expired aliases get deleted?

**Extended questions:**
- The same codename means opposite things in two product lines. Without forcing the user to pick a product, how does retrieval avoid crossing the streams?
- If you only add definition chunks and do not touch the encoder, is that enough for queries that never spell out the term?
- How do jargon-slice and general-slice recall together prove the glossary did not pin everyday language?

**Reasoning:** Constraints are “never drop the raw token” and “do not globally unique-sense a word.” Order the tools by failure type: lexical exactness, controlled contextual expansion, then the vector space. Accept with the jargon/general recall pair so common words stay unpinned.

## RAG-016. First-stage Recall@50 already meets the bar, but the generator keeps treating the 12th-ranked passage as evidence. Where do you insert a more expensive scorer, and how many candidates do you feed it?

**Answer:** Insert it after first-stage retrieval and before prompt assembly. Score those ~50 passages with a cross-encoder or similar query–passage pairwise model, and promote the 4–8 spans that can actually support the answer to earlier (or otherwise easier-to-use) prompt positions. Candidate count should be the first-stage k that already covers gold; raising it further yields shrinking gains and roughly linear extra latency. If the passage still misses the final m slots after rerank, inspect assembly truncation or generation next—do not start by enlarging the generator.

**Deep dive:** First stage (bi-encoder, BM25, or a hybrid) optimizes recall; its score is not aligned with “can this span answer the question,” so useful spans often sit outside the top ten. Rerank works on a short list and grows roughly linearly with k. At k = 50, if each forward pass is on the order of 10–20 ms, the stage can reach 0.5–1 s unless you batch or cut k to the smallest value that still covers gold. If the generator can spare only about 2k tokens of evidence, 4–8 medium chunks already fill it; stuffing all 50 is expensive and dilutive. The objective of this layer is therefore precision@m (m = spans that enter the prompt), not another Recall@50 point. Log first-stage rank and rerank rank together so you can see moves such as 12 → 2. If the answer still ignores the span after rerank, the bottleneck has left retrieval.

**Follow-ups:** When k goes from 20 to 100, does rerank latency blow up first or does end-to-end accuracy saturate first? If the reranker and the embedder were trained on similar data, will they share the same near-miss passages? Is it sound to rerank only the mid-ranked band and leave the very top and bottom untouched?

**Extended questions:**
- How do you use gold-span rank displacement to prove this layer is worth its cost, rather than user thumbs-up?
- What does ranking prompt spans by rerank score versus original reading order do to citation completeness?
- If the reranker crushes all 50 scores into a tight clump, is the model broken or does the query lack distinguishing information?

**Reasoning:** Recall is already sufficient; prompt slots are not. Add a pairwise scorer that is responsible only for precision on the short list, and pin k at the smallest first-stage value that covers gold. Accept the layer with rank displacement and precision@m so the generator window is not reused as a retriever.

## RAG-017. You may spend only about 80 ms to rerank 64 candidates. Cross-encoder, token-level late interaction, or LLM listwise rerank: which one do you bet on, and why not the other two?

**Answer:** Under an 80 ms / 64-span budget, prefer late-interaction scoring with document representations already computed; a small cross-encoder can also fit if you have a GPU and a batched graph. Do not send all 64 spans to a large model for listwise rerank: that pays long-context encode plus generation time and often has position bias, and it will not finish in 80 ms. Late interaction wins this budget because the expensive encode happened at index time; online work is one query encode and 64 MaxSim scores.

**Deep dive:** A cross-encoder concatenates query and each candidate and runs full self-attention. Pairwise interaction is richest and accuracy is usually highest, but the document side cannot be precomputed and time scales roughly with k. A mid-size model batched on GPU may land in tens of milliseconds for 64 pairs; on CPU it usually misses the budget. Late interaction already stores document token vectors, so the 80 ms mostly covers query encoding and short-list MaxSim, typically a bit below a good same-size cross-encoder. LLM rerank is pointwise, pairwise, or listwise. Listwise is strongest on instructions such as “find the span that contradicts this clause,” but 64 spans can be tens of thousands of input tokens, one call often takes hundreds of milliseconds to seconds, and earlier positions are more likely to be chosen. A common compromise is a cascade: late interaction or a small cross-encoder reduces 64 to 8 inside the budget, and only hard instruction queries spend extra on an LLM. Accept on post-rerank nDCG@8 and end-to-end accuracy while watching p95. If the top span changes and the answer does not, the bottleneck is not rerank.

**Follow-ups:** Is 64 a measured first-stage k or a guess, and how do you cut it using gold coverage? Without a GPU, does the 80 ms budget still hold for a cross-encoder? If the LLM reranks only the first 8, can it scramble an already-correct order?

**Extended questions:**
- If listwise rerank cannot fit 64 spans and you slide a window, how do you merge ranks across windows?
- When fine-tuning a cross-encoder on business data, do you mine hard negatives from first-stage top-100 or sample them at random?
- On short identifier queries versus long comparison queries, can the three rerankers swap relative rank?

**Reasoning:** Rewrite the constraint as “a new ranking of 64 spans in 80 ms,” then drop options by whether document representations can be precomputed and whether scoring needs autoregressive generation. The LLM exits on latency and position bias; the cross-encoder depends on hardware and model size; late interaction fits the budget. Keep a cascade so not every query is glued to one reranker.

## RAG-018. Every paragraph is tagged [3], but the click opens a different page. How do you pin citations to the span that actually supports the sentence, across ingest, generation, and a post-check?

**Answer:** A citation is a verifiable link from a claim to a source span, not a block number the model sprinkled on. At ingest, every chunk carries a stable id, document version, page, and character offsets. At generation, quote first and answer second; block ids may only refer to ids present in this prompt. Afterward, entailment-check each claim and drop or abstain on a mismatch.

**Deep dive:** Typical failures: the model only learned “answers need footnotes” and cycles [1][2][3]; prompt numbering is shifted relative to chunks; the cited document is related but not the sentence that supports this number. Three pins. Ingest stores an immutable `chunk_id` plus `doc_id, version, page, start, end, text`, and the prompt uses `[#c17]` plus a short excerpt instead of 1, 2, 3. Generation is two-step—emit claim, quote, chunk_id, then polish—or decoding restricts citation tokens to the allowed set. The post-check tests containment or entailment of each claim in its chunk; keep numbers and negations lexically tight—18% should not hang off a span that only says “about twenty percent” unless the rubric allows that paraphrase. A click must open the source at the offset and highlight it, not the document cover. Score click-is-supporting, coverage of key claims, and dangling ids that were never in the prompt. In compliance settings, a wrong cite is often worse than no cite.

**Follow-ups:** How you represent two spans that jointly support one conclusion; how you degrade when the UI can only show a document-level link; what happens if the model emits a chunk_id that was not in the prompt.

**Extended questions:**
- Write a minimal structure that maps a claim to a source span and mark which step validates it.
- How do you tell “right document, wrong paragraph” from “wrong document”?
- While streaming, the post-check is not done yet. What do you show the user first?

**Reasoning:** The goal is traceable, automatically rejectable attribution. Models will not naturally align block numbers. Close the id set to what the prompt contains, cut dangling cites with entailment, and define correctness as “a click lands on the sentence that supports that claim.”

## RAG-019. In eval you move the only supporting passage from slot 1 to slot 6 and accuracy drops. Is retrieval broken, or did the model skip the middle?

**Answer:** If the retrieved set is unchanged and only the concatenation order changes, treat it as position bias, not a miss. Many decoder-only models use the beginning of a long prompt and the tail next to the question much more than the middle; a relevant span parked in between is easy to skip. That U-shape is what people call “lost in the middle.” A longer window does not flatten it. Cut noise, rerank, pin high-value evidence at the two ends, and force the model to name the sentences it used.

**Deep dive:** A RAG prompt is usually system text + retrieved chunks in order + the user question. Attention is not uniform over a long sequence: early chunks get a primacy boost, chunks next to the question get a recency boost. With 10 chunks of roughly 300–500 tokens, slots 5–7 sit 1–2K tokens from either end. That is orthogonal to recall: context recall can look fine while the model never uses the gold span. Locate it with a position sweep: keep the same top-k, insert the gold chunk at index 0, k/2, and k−1; a U-shaped accuracy curve is the signature. Mitigations, in order: a cross-encoder that promotes the real 3–5 chunks to the front or next to the question; drop low-score noise so the packed context shrinks from, for example, 4K to 1.5K tokens; generate as “quote then answer,” or NLI-check the claims. Raising the window from 8K to 32K mostly lengthens the neglected middle; the U often remains.

**Follow-ups:** If the gold chunk also fails at the end, is that truncation, table token blow-up, or the system prompt eating the tail; how you attribute a sweep that is run together with reranking; whether streaming makes the model lean even more on the first chunks it reads.

**Extended questions:**
- How do you build a regression set that only permutes chunk order, without teaching the model to read numbering instead of text?
- If you copy the same evidence to both the front and the middle and accuracy recovers, what does that prove and what does it not?
- If you must keep ~20 chunks for multi-hop, how do you read them in passes instead of dumping all of them into the middle band?

**Reasoning:** Separate “evidence never entered the prompt” from “evidence entered and was unused.” The constraint is a fixed set and a changing order. A U-curve means position preference: shorten the effective context, move key chunks to the ends, and make middle evidence unskippable via citations.

## RAG-020. How do you keep evidence from drifting in sense after you compact retrieved passages?
**Answer:** Preserve original positions, entities, numbers, negations, and conditions; distinguish extractive compression from generative summarization. Evaluate whether answers and citations stay consistent before and after compression, and reread the original when needed.
**Deep dive:** Compressed text must not become a new source of authority; citations should land on the original version.
**Follow-ups:** How do you construct regression samples where a number or a negation is wrongly dropped? Use minimal paired texts that contain units, ranges, exceptions, and negated conditions, and check whether the conclusion changes after compression.
**Extended questions:**
- When one conclusion in the compressed text depends on multiple original snippets, how should you store the source mapping?
- If the answers before and after compression agree but both are wrong, why do you still need original-text-level fact checking?
**Reasoning:** Obtain token savings with evidence that remains traceable, rather than replacing facts with an unverifiable summary.

## RAG-021. The answer is already generated with citations. How do you add a post-generation claim check—splitting atomic claims, testing each against its cited chunk with NLI or an LLM judge—estimate the latency and cost, and choose regenerate, drop the claim, or abstain on failure?

**Answer:** Split the finished answer into atomic claims (one predicate, one slot). Bind each claim to its own cited chunk, or to all cited chunks if citations are not per-sentence. Check numbers, dates, and IDs by normalized surface match; check qualitative claims with an NLI model or a short judge call, producing entail / contradict / neutral. Run this before the user sees a final draft. Contradiction, or a neutral on a must-have slot, means abstain or drop that sentence; allow at most one regenerate, then stop. Budget as claims × cost per check: a small NLI model can batch into tens of milliseconds, while a sequential LLM judge per claim adds another full generation-sized delay.

**Deep dive:** Do not split on periods alone: “30 days and the customer pays shipping” is two claims. A short prompt can emit `{slot, value, cite_id, text}`; run a regex pass so numbers and proper names are not dropped. The judge must see premise = cited chunk text and hypothesis = claim text only—no dialogue, no parametric memory. A hybrid is sturdier: amounts, dates, ticket IDs, and percents go through unit-normalized equality first (match → entail, mismatch → neutral or contradict); remaining qualitative sentences go to an entailment classifier (for example a 100M–1B-parameter NLI model) or one short judge completion. Magnitudes: if the original answer took about 2s, claim-split is one 100–300ms call, and 4–6 claims batched through NLI often cost tens of milliseconds; five sequential generative judges at about 300ms each add about 1.5s and will break P95. Parallelize when you can. If only about 400ms of P95 remains, verify high-risk slots only, or skip the gate on this request and sample offline. Branch on slot risk: contradiction on price, permission, dose, or a compliance number → abstain or escalate; do not silently delete the sentence so the rest looks reviewed. A decorative clause that is neutral → drop that sentence or say the materials do not state it. Several neutrals on slots the question actually asked → abstain. Regeneration is one round: paste the failed claims and the allowed chunks back, forbid numbers not in those chunks, verify again, and stop if it still fails so you do not loop check–rewrite. If tokens already streamed, drop/abstain must send a correction or retract; changing only the log is not enough.

**Follow-ups:** When the judge is the same size as the answer model, how do you stop it from calling “same topic” entailment; if a conclusion needs two chunks together, do you concatenate premises or force the generator to emit two claims each tied to one chunk; after a wrong number has already streamed, is a retract or a patch safer for a support desk.

**Extended questions:**
- If you surface-check only numbers and dates and skip NLI on qualitative sentences, what unsupported wording still gets through, and what share of claims is that?
- If you move the check from a synchronous gate to async sampling, how do you book the P95 you saved against the finals that shipped unverified, using the same week of logs?
- On amount questions, how do one regenerate, drop-the-sentence, and full abstain compare on false blocks versus misses?

**Reasoning:** The goal is that every checkable claim in the final draft is supported by its cited chunk. The constraint is that generation already spent most of the budget, so verification cannot clone an equally expensive generation path, and a failure must not invite another unconstrained draft. Split atomic claims, use surface match for numbers and a short entailment check for the rest, and hard-branch among one regenerate, drop, and abstain by slot risk. Accept the design only if claims × per-check latency fits the leftover P95.

## RAG-022. The final number needs a project code from document A, then that code must be looked up in document B’s budget. Why does one top-k pass often miss both ends, and how should the retrieval schedule run?

**Answer:** The user question usually does not contain that code, and B’s query terms do not exist until A has been read. One top-k on the original sentence may bring A in; B often stays out because it shares little wording or semantics with the question. Retrieve first, extract the intermediate entity, rewrite a second query with that entity, and assemble evidence hop by hop. Raising k from 10 to 50 does not create query terms that do not exist yet.

**Deep dive:** Single-hop retrieval matches the user’s sentence. Second-hop evidence is often a code table or a budget row with low lexical overlap with the question. A workable schedule: decompose into ordered sub-questions (“what is the official code?”, then “budget row for that code”); or iterate retrieve(q0) → extract entity → retrieve(entity + original need) → stop when evidence is enough or the step budget is spent. Each hop can use a smaller k, for example 5–10, because the query is now specific. Do not rely on “stuff more documents and let the model connect them”: if B was never retrieved, it is not in the context. Evaluate per hop: was A found, was the entity extracted correctly, was B found, is the final number supported by B? End-to-end correctness mixes extraction errors with retrieval errors. If two hops still lack evidence, refuse or ask the user to confirm the code; do not let parametric memory invent a tidy budget.

**Follow-ups:** If the intermediate entity is wrong, how do you stop hop two from confidently retrieving another project’s budget? How do you split the latency budget across hops, and if hop one times out do you cancel or continue with a partial entity? Beyond three hops, do you keep a linear schedule or do a bounded walk over entity relations?

**Extended questions:**
- How do you write eval items that require A then B so the model cannot skip hop two with parametric knowledge?
- For hop two, is a rewritten natural-language query or an equality filter on the code less likely to drift to a same-name project?
- When several documents enter the prompt, how do you force separate citations of A and B instead of one unsourced blended sentence?

**Reasoning:** Draw the evidence dependency: B’s lookup key is an output of A, not a token in the user question. That rules out a larger k and a single larger context dump. Design a minimum two-step schedule and split extraction from retrieval with per-hop metrics. If the chain is incomplete, refuse so hop two cannot emit a precise number attached to the wrong code.

## RAG-023. A 280-page regulatory handbook has a clean table of contents. Do you chunk it into a vector index, or let the model walk the heading tree?

**Answer:** When the TOC is trustworthy, you have a few long documents, and the question depends on “which chapter,” tree navigation (little or no vector search) is often more accurate: the model reads titles and abstracts, decides which branch to open, and only then reads the leaf. For a huge pile of short, unstructured docs, or when you need tens of milliseconds, a vector index stays the default. A common hybrid is keyword/vector to one book or chapter, then a tree walk inside it.

**Deep dive:** Offline you build hierarchical nodes: title, short abstract, child pointers, leaves pointing at source paragraphs. Online you start at the root; the model sees the current directory and picks 1–3 children until a leaf. The signal is reasoning over titles—“indemnity caps” live under Claims, not Definitions—not cosine neighbors, so you avoid near-matches that sound alike and sit in the wrong chapter. The cost is one model call per level; four levels of a few hundred milliseconds plus reading the leaf is often seconds, which will not carry very high QPS. Tree quality is the ceiling: without a TOC you can induce levels from headings or a small model, but a wrong parent prunes the gold branch, so you need a backtrack budget. Unlike trees that embed summaries and then vector-select nodes, this path can be pure model choice with no embedding model or vector store. Hybrid: retrieve a few candidate books with vectors, then disable vectors inside the book and walk the TOC. Eval should score “did we pass through the required section,” not only final correctness.

**Follow-ups:** How many times you backtrack after a wrong branch before stopping; whether the tree is usable when abstracts are stale and the source was revised; how the navigation log becomes reviewable evidence.

**Extended questions:**
- The handbook is split into 200 pages with no unified TOC. Do you still refuse vectors?
- How do you grade “must-visit sections” instead of only the final answer?
- If each level scores a handful of candidates instead of picking one, how do you trade latency?

**Reasoning:** The constraints are structure, corpus size, and latency. Clean structure, few books, humans would use the TOC → walk the tree. Many docs, weak structure, need throughput → vectors. Ask how a person would flip through the book, then pick the index shape.

## RAG-024. Half the traffic is chitchat and half needs policy lookup. Is it more stable to let the model emit a “retrieve now” signal than to retrieve every time?

**Answer:** Stability comes from a testable retrieval gate, not from sounding smarter. Self-RAG-style systems make the model emit retrieve/critique markers on the generation trace: decide whether this step needs the corpus, then whether each passage is relevant, then whether the continuation is supported. Chitchat and pure common sense can skip retrieval; policy, numbers, and time-sensitive facts must not. Without labeled markers or a rule fallback, the model will skip risky questions. Default retrieve-and-ignore-useless-passages is safer than an ungrounded gate.

**Deep dive:** The control flow is not one retrieve for every request. The model first emits a retrieve control token (or a small classifier fires); only then is the index queried. Each passage is marked relevant or not. After each generated span, a support marker decides rewrite versus another retrieve. That is different from an agent tool loop: the decision lives as special tokens on one generation trace and usually needs training on marked data, or at least very hard instruction constraints. A common engineering compromise is a light gate on cheap features—policy names, dates, ticket ids, amounts—more controllable than letting the base model improvise. Miss cost is asymmetric: a wrong price is far more expensive than an extra retrieve, so high-risk intents always retrieve and only obvious chitchat is short-circuited. Score the gate’s precision/recall, the accident rate of “should have retrieved and did not,” and context recall after a trigger. Latency savings count only if the gate actually skips retrieves and the accident rate is acceptable.

**Follow-ups:** What you do when the classifier lets through “sounds like chitchat, actually asks the indemnity cap”; whether an untuned model can emit special tokens from a prompt alone; whether unsupported retrieval should force a continuation or an abstention.

**Extended questions:**
- Name at least three question templates that must always retrieve and must not be skippable.
- How would you estimate, from logs, the latency saved by one skipped retrieve and the business cost of one miss?
- Mid-generation the evidence is insufficient: retrieve again, rewrite the query, or abstain?

**Reasoning:** Retrieve when you must, skip when you can. A miss costs more than an extra lookup, so hard-require retrieval on the high-risk set and only then hand the rest to a model or classifier gate. Do not ship conditional retrieval without trigger metrics.

## RAG-025. A troubleshooting assistant can rewrite the query and retrieve again; after launch it averages 3.4 hops and keeps hitting the same chunks. How do you write the stop rule, hop cap, and loop detector so each extra hop's latency and cost stay inside the budget?

**Answer:** Before starting a hop, ask whether you can still afford it and whether it can still add information; stop if either answer is no—do not wait for the model to say it is done. Back out the hard cap from the interactive SLA: if you reserve about 1.5s for the final generate and a hop costs about 0.7–1.2s, production gets two hops and never more than three. Detect loops with a normalized query fingerprint, near-duplicate query embeddings, and whether the new top-k chunk IDs were already seen—not exact string equality. A 3.4 average means the stop rule is barely firing; kill the empty spins before you discuss adding hops.

**Deep dive:** A hop’s bill is usually not one ANN call: rewrite 150–250ms, retrieve 50–100ms, rerank another 300–600ms if you use it, plus a 150–300ms “is this enough” check—about 0.7–1.2s wall time and another short generation’s tokens. A 3.4 mean is about 2.4 extra hops, which lifts a ~2s single-hop path into the 4–5s range and adds cost almost linearly. Evaluate the stop predicate before you issue the next retrieve: hops ≥ `H_max`; remaining latency `<` the estimated hop cost (recent-window p50 plus slack); the new query is a near-duplicate of any earlier hop; the new chunk_id set is a subset of the seen set, or unique new chunks are fewer than one; this hop is empty or the best score is below a floor; or sufficiency is true. Sufficiency is “required slots for this question type are filled from current evidence” (error code, service name, last change), not self-reported confidence. Loop detection is three layers: a query hash after NFKC, lowercasing, and whitespace collapse; query-embedding cosine against history above, for example, 0.92; and `new_ids ⊆ seen_ids`. Exact string dedup misses spins like “circuit-breaker threshold payment” versus “payment gateway breaker how to set.” Do not copy a research-agent `H_max` of 6–8: subtract final-generate time from the SLA, then divide by per-hop cost. Sufficiency should fire before the cap, or the system always runs to the wall. Boundary: a totally empty first hop is not “enough”—allow one rewrite, then stop. If slots are still open but the budget is gone, return “what we found + which slots are missing”; do not buy one more hop as a gamble.

**Follow-ups:** If you drop the near-dup threshold from 0.92 to 0.85, which legitimate second-hop rewrites (for example adding a service name) get killed; when the sufficiency judge is the same size as the answer model, how do you stop early-stop and never-enough; if rerank is expensive, may hop two skip it and only lexically fill an entity.

**Extended questions:**
- Write the stop-function order: budget, hop count, query near-dup, no new chunks, slots filled. Which predicate must run first?
- How do you estimate, from a week of traces, the mean unique chunks added by one more hop, and pick the hop where that gain falls near zero as the default `H_max`?
- When the user follows with “what about prod,” do you reset the hop counter as a new question or keep the seen-chunk set—what does each choice pay for twice?

**Reasoning:** Extra hops should happen only when they can still buy unseen evidence and you can still finish generation on time. The constraint is that each hop costs about one short RAG in wall clock and tokens, and repeated chunks have no value. Hard-stop on budget and `H_max`, catch loops with query near-dups plus chunk-set gain, soft-stop when slots are filled, and accept the policy on mean hops and unique chunks per hop—not on whether the model still wants another search.

## RAG-026. A user wants last year’s contractor leave policy compared with this year’s, and one retrieve-then-generate pass keeps missing clauses. When should the model be allowed to decide the next retrieval?

**Answer:** When a single top-k list cannot hold both years, or the second policy’s query terms are unknown until the first policy is read, hand “retrieve again / with what query / from which source” to a budgeted loop. After reading this year’s clauses, the model should issue a second search for last year’s version or a specific clause id, then synthesize. If the question is really a single-document FAQ, do not open that loop or you double latency and cost for nothing.

**Deep dive:** A fixed pipeline is retrieve(q, k) → generate. Here the retriever is a tool: each step the model emits an action (search / stop) and the system writes the observation back. This item needs at least two hops—find this year’s contractor-leave section, extract clause names to compare, then search last year with those names plus the year. Hard caps are required: max steps (for example 3–6), max retrievals, total token and latency budget, an allowlisted tool set, and a log of each query and hit IDs. Typical failure is rewriting the same query in a tight loop or fetching an already-read chunk; block that with query dedup and a set of seen chunk ids. This is multi-step tool scheduling, not a one-shot “should I retrieve at all” gate. Two or three rounds turn one retrieval plus one generation into several model calls, so enable the loop only for compare / cross-version / cross-system questions.

**Follow-ups:** How do you automatically route a request to fixed single-hop versus looped retrieval? If the step budget expires and one year is still missing, do you refuse, answer only the known year, or scan more documents? How do you stop the model from inventing a last-year clause id and searching for it?

**Extended questions:**
- What counts as success on a comparison question: complete clause-level alignment, or a list that merely looks plausible?
- Should the retrieval tool expose a “filter by version” argument instead of making the model stuff the year into the query string?
- How do you compress a looped-retrieval trace so a reviewer can see why two extra calls were spent?

**Reasoning:** Ask whether the failure is “one query string cannot cover two evidence sets.” If yes, a larger k still cannot invent next year’s query terms. Turn retrieval into a tool loop with stop conditions, and lock cost and wandering with step limits and an allowlist. Keep simple questions on a single hop so the loop is not the default path.

## RAG-027. A user asks who reports indirectly to the CFO and also handled vendor X’s contract. Vector top-k usually hits only one hop. When is it worth materializing a graph?

**Answer:** When the answer depends on an explicit entity–relation–entity path, not on a paragraph happening to be close to the question in embedding space, a graph beats flat vectors. Reporting lines, vendor–contract–owner, part–substitute: a single similarity query stops at a paragraph that mentions the CFO or one that mentions X and never walks the joint. If the question is “how do I fill the contract template,” a graph is extra cost and extraction noise; vectors plus query decomposition are enough.

**Deep dive:** Offline you extract triples with source spans, disambiguate entities, and load a knowledge graph; community or subgraph summaries can be a second retrieval unit. Online you link the question to seed entities (CFO, vendor X), walk 1–3 typed hops, and give the generator the source sentences on that path—not the node labels alone. Iterative vector search can also multi-hop (contracts for X, then the owner in the org doc), but the second hop depends on wording. A graph materializes the relations so the query is a typed walk. The main cost is extraction and entity resolution: two people named Li Wei, or a “cc:” turned into a reports-to edge, make later walks systematically wrong and confidently complete. Document updates must add and delete edges, not just re-embed. A practical bar: a large slice of the eval set requires combining at least two non-adjacent documents, and entity types are stable. Otherwise start with decomposition plus iterative retrieval and keep the graph for when relation density is truly high.

**Follow-ups:** Why a bad entity link makes a graph walk look more confidently wrong than a vector miss; whether community summaries or raw triples are the generation evidence; whether a citation should point at a node or at the sentence used to extract the edge.

**Extended questions:**
- For reporting lines and contracts from two heterogeneous sources, how do you design a schema that does not record a cc as reports-to?
- At what extraction accuracy do you shut off walks and fall back to iterative retrieval?
- How do you score “path right, attached source span wrong”?

**Reasoning:** Ask whether the question is a typed relation composition. If one-hop lexical/semantic match covers it, skip the graph. If you must walk edges, materialize them and keep source sentences as evidence. Accept on path hit rate, not surface similarity.

## RAG-028. The business asks for quarterly gross margin by division. Do you embed each table row, or let the model write read-only SQL? What routes the traffic?

**Answer:** Cell questions that filter, group, sort, or aggregate should go through constrained text-to-SQL or fixed query templates. Do not flatten the whole table into row vectors and hope similarity hits the right number. Vectors are for finding which table or paragraph defines the margin convention. The number itself must come from a query result or from a structured snippet that still has headers and units. The router is simple: comparison, time range, or grouping → the query engine; “what does gross margin mean” → document RAG.

**Deep dive:** Row embeddings fail in a predictable way: a row becomes “East 2024Q3 gross margin 18%,” and “which division is lowest” is not a nearest-neighbor operation, so top-k returns similarly worded rows instead of the min. The SQL path is schema linking (tables, columns, enums) → a read-only SELECT → a sandbox → the result table (cap it, e.g. 50 rows) plus the convention note into the prompt. Hard rules: read-only credentials, column-level grants, no multi-statement, timeouts, result-size caps. If the numbers still live in a document table, keep headers, units, and cell coordinates, serialize to Markdown or CSV, and retrieve on “table title + header,” not headerless cells. The generator must not mentally re-total numbers that are not in the result. Score execution correctness and numeric rewriting, not whether the sentence sounds fluent.

**Follow-ups:** How you avoid picking the wrong same-named column in a large schema; who wins when SQL is right and the convention doc is stale; what you fall back to when merged cells break the parser.

**Extended questions:**
- Write tool constraints that allow only allowlisted views and reject multi-statement or cross-tenant joins.
- One question needs both the definition and this period’s number. How do you pack both paths, and who wins a conflict?
- How do you catch the model turning 18% into “about twenty percent”?

**Reasoning:** Numeric questions must be executable and reproducible. Similarity search does not carry comparison or aggregation. Classify intent: definitions through documents, computation through a query engine, then pin generation to the result table so the model cannot do the arithmetic in its head.

## RAG-029. In a two-column financial PDF, cross-page tables and running headers are sucked into the body and numbers no longer line up with rows. Which errors do layout parsing, table reconstruction, and kept coordinates each fix, and do scans need their own pipeline?

**Answer:** Reading across both columns in one sweep is a reading-order failure: restore left column then right with layout analysis, not a better embedding. A table whose header lives on the previous page must be reconstructed and stitched by header, not stored as two unrelated chunks. Running headers that repeat every page should be dropped by clustering coordinates, or they pollute chunks and lexical weights. Coordinates make citations land on a page and cell and make header stripping reproducible. Scans go through OCR plus confidence and quarantine low-confidence pages; born-digital PDFs use text objects and fonts. The error modes differ, so the pipelines should.

**Deep dive:** The break is in parsing, not vectors. Naive extractors follow paint order or sweep by x, so two columns become half-sentences glued together and tables become a bag like “revenue 12.3 cost 9.1.” The model can only be faithful to a wrong table. A layout model emits boxes and order for paragraphs, headings, tables, and headers. Table structure emits the cell grid, row/column spans, and header hierarchy, then serializes to headed Markdown or row records. Coordinates (1) find y-periodic headers/footers, (2) align the same table’s column x-ranges across pages, and (3) support page+box citations. Scans have no reliable text objects; OCR confuses “.” and “,” and thousands separators, so you need character-level confidence and must not treat low-confidence digits as facts. Edge cases: a wide table that spans columns, a rotated scan, a footnote that shares a number with the table—all need sampled human checks. Do not gate on overall similarity between extracted text and the source; the page can look alike while every cell is already wrong.

**Follow-ups:** If reading order is wrong but table structure is right, which do you fix first? Without a repeated header, what evidence says two pages are the same table? At OCR confidence 0.8 on an amount, does the product show it with uncertainty or refuse?

**Extended questions:**
- How does whole-page OCR then chunking differ from “layout first, OCR each region” in fixing two-column mis-order?
- After tables become row records, how do you keep total rows and comment rows from being retrieved as ordinary data?
- How do you use cell-level checks (for example 20 financial pages) instead of full-text overlap to decide whether a parser may enter the index?

**Reasoning:** The goal is that numbers align with row, column, and page; the constraint is that PDF paint order is not reading order. Pick the tool by error class: order via layout, cells via table structure, headers and citations via coordinates, scans via a separate OCR pipe. Accept with cell accuracy and cross-column error rate, not full-text similarity.

## RAG-030. The critical steps in an ops manual live in screenshots and parameter tables, so a text-only index answers wrong. What does each encoding drop — image-as-caption, image-as-visual-vector, table-as-cell-records — and how do you mix them?

**Answer:** Caption/OCR gets you into the existing text ANN, but it drops UI state that was never written down (a red box, a button locus) and any OCR typo; without pixels at generation time you also cannot copy a parameter off the image. A visual vector retrieves “a similar screenshot” but cannot match a text question if there is no readable string, and citations rarely land on a cell. Row records that repeat the header fit “what is this service’s timeout” and break merged cells, footnotes, and totals. Production usually keeps all three: image bytes plus boxes, table Markdown plus row chunks, union retrieval, and answers that point at a page and region.

**Deep dive:** Ingest forks by modality. Screenshots: OCR spans + optional caption + `page, bbox`; if generation is multimodal, pass the original image after a hit, otherwise only OCR. Tables: structure first, then (1) a whole-table Markdown chunk to keep relationships and (2) one chunk per row with field names repeated so “8 / 16 / 32” never detaches from the column. Retrieve with text vectors, image vectors, or both, then aggregate by document. Edge cases: flattening a table into a bag of tokens aligns numbers to the wrong service; a vision-only index misses a purely textual timeout question. Citations need page and coordinates so a human can check the PDF. Do not confuse this with “stuff the whole manual into a long context”—structure is lost at parse time, not because the window was short.

**Follow-ups:** Without a multimodal generator, will you answer parameter questions from OCR alone, and at what typo rate do you refuse? How do you dedup a row chunk and a whole-table chunk that both hit? If image and text vectors do not share a space, do you search separately and fuse?

**Extended questions:**
- One figure contains both an architecture diagram and a parameter table. Do you cut two objects or one multimodal chunk?
- How do you encode two-level merged headers in a row record so “prod / timeout” does not bind to “staging / timeout”?
- How would a set of “parameter questions must hit a row or an image region” show that you did not merely add noisy captions?

**Reasoning:** The goal is to locate a step or parameter on a figure or a row, not to retrieve a vague paragraph. Compare the structure each encoding drops, then keep a union plus citations instead of one flattened text. Accept with row/region hit rate and misalignment rate on parameter questions.

## RAG-031. The return-policy PDF was replaced yesterday; today retrieval still ranks the old file first. Incremental upsert, tombstone deletes, or recency down-weighting—which layer do you fix first?

**Answer:** Fix delete propagation first. Old chunks must leave the serving index or be atomically swapped for the new version; inserting new vectors is not enough. Upsert only guarantees that the new rows appear. If old ids remain, the old text is still similar to the new question and will keep winning. Recency down-weighting is a fuse while two versions briefly coexist, not a substitute for delete.

**Deep dive:** The write path is change detection (content hash, source updated-at, file etag) → write new chunks and batch-delete old ids, or flip an alias → cut read traffic only after replicas are consistent. Tombstones must land in both the vector index and the lexical index, or you get “vectors gone, BM25 still there.” Chunk metadata should keep source_id, content_hash, effective time, and what it supersedes. On the query side, policy intents get an effective-window filter; expired docs score to zero. This is convergence of the live index with the source, not historical snapshots or as-of replay. Split the SLA: index visibility (often minutes) versus the longest window an old version can still be retrieved. Watch documents whose source hash moved but the index hash did not, delete lag, and the version date cited in answers. Hot-query cache keys must include the index version, or a successful delete still serves the old answer.

**Follow-ups:** Same URL, new bytes, crawler hash cache stale—what then; how replicas avoid a split-brain of half old and half new; how cache TTL lines up with the index visibility SLA.

**Extended questions:**
- Design one update that replaces 10 old chunks with 12 new ones so no in-between state is searchable.
- If old clauses must not be cited within ten minutes of a change, which metrics prove it?
- Only one section changed. What are the risks of full re-embed versus local re-chunk?

**Reasoning:** The live index should have exactly the source’s current version. Expiry does not lower cosine similarity by itself. Make delete/replace atomic across both indexes, use time filters for the propagation window, and only then talk about down-weighting.

## RAG-032. The auth header changed last week, but the system still cites the old API page. How do freshness metrics separate sync lag, undeleted old chunks, and a generator that prefers the longer old text?

**Answer:** The three leave different traces. Sync lag: the source’s mtime is newer than the index time, the store still has the old hash, and no new chunk exists. Undeleted old chunk: both hashes are searchable and the older, longer one often scores higher. Generator preference: both versions entered the context and the model cited the more complete old paragraph. Use the `source time − index time` histogram for lag, “same doc_id with multiple live hashes” for ghosts, and “which rev was cited when both were selected” for generation bias. Fix with faster sync, tombstones on old hashes, ranking/prompts that prefer the latest dated source; for auth and prices, query a live system instead of the static corpus when needed.

**Deep dive:** Each chunk stores `source_mtime`, `indexed_at`, `content_hash`, `rev`. Lag: alert when `now - indexed_at` exceeds the SLA (for example 24 h on high-churn API docs). More machines do not help; the connector and change notifications do. Ghosts: an old hash under the same `doc_id` was never tombstoned, so retrieval looks “updated” while top-1 is still the old body; delete fan-out must cover every shard and cache. Generator preference: even if the reranker puts the new chunk first, a long old passage can win attention—up-weight the new chunk, truncate undated old text, and require a source date on the answer. Edge cases: wrong source clocks invert recency; a page that only changed navigation yesterday has a fresh mtime and the same auth section—use section-level hashes. This is not compliance versioning: here the default online answer must match current source text, not a historical replay.

**Follow-ups:** Is the SLA per document type or one global hour? If the live system fails, do you serve the dated old chunk with a warning or fail closed? When the cache is keyed by query text, how do you actively invalidate it after a source update?

**Extended questions:**
- Why does monitoring only mean index lag still miss “old chunks you cannot delete”?
- The new chunk is short and the old one is long. If the reranker has no time feature, what do you add?
- How would a weekly report of lag, ghost rate, and dual-version citation share show that stale answers are falling?

**Reasoning:** The default answer should match the current source. Use per-chunk time and hashes to split ingest, delete, and generation, and fix each layer instead of “sync again.” Accept with the three metrics and dated citations.

## RAG-033. Compliance requires every Q&A to replay the document versions it cited, while policies are revised every hour in daytime. How do you combine index snapshots with chunk versions so each revision does not re-embed the whole corpus?

**Answer:** Put reproducibility on a version stamp on the answer; put frequent edits on dirty-hash re-embeds. Persist `snapshot_id` plus each cited `(doc_id, rev, content_hash)`; a replay uses that snapshot or a point-in-time filter on `effective_from/to`. Hourly revisions write an incremental segment only for changed hashes; unchanged chunks keep their vectors. Freeze a daily read-only snapshot for audit rollback. Full-corpus re-embed happens when the embedding model or the hash function changes. Compliance needs a rebuildable citation set, not an hourly copy of every vector.

**Deep dive:** Chunks are immutable versions: a paragraph edit yields a new hash and vector; the old one gets an expiry instead of in-place overwrite—overwrite makes yesterday’s answer inexplicable. Incremental segments behave like an LSM: hot writes go to a new segment, queries read hot then the daily snapshot, the newer rev wins for the same `chunk_id`, deletes are tombstones. If about 1% of documents change per hour, re-embed cost is about one percent of a full pass. Historical lookup at time t sees only `effective_from ≤ t < effective_to`. Edge cases: too many snapshots explode storage—keep hot segments for hours to a day and daily snapshots for the retention period; ACL and body must version together or a replay can surface a chunk that was not visible then. This is not the same as stale answers: even when the latest answer is right, you must be able to walk back into any logged turn.

**Follow-ups:** If the embedding model has been replaced, does replay use old vectors or re-embed the then-text with the new model? If a cross-snapshot incremental segment is corrupt, how do you still produce an audit view? If citations store only `doc_id` and not the hash, which disputes can you no longer settle?

**Extended questions:**
- A document changes three times in one hour. Should a query see the last write or the freeze at hour end?
- How do you estimate storage for “hot segments + 30 daily snapshots” versus copying the full index every hour?
- If Q&A logs lose the hash field, how do you show that compliance replay is no longer possible?

**Reasoning:** Constraints are point-in-time replay and a short change window. Make chunks immutable hashed versions and the index a hot increment plus a cold snapshot; encode only dirty hashes. Accept when a replay hits the same hashes and hourly re-embed fraction stays near the dirty rate.

## RAG-034. The wiki says the reimbursement cap is 3000; a year-old email says 5000; the model averages them to 4000. When do authority order, timestamps, and explicit conflict marking each apply, and when must you refuse?

**Answer:** 4000 exists in no source—it is a fabricated midpoint and must be banned. Authority order fits when ranks are clear and the higher source is still in force: a published policy outranks an unfiled email. Timestamps fit a revision chain at the same authority level: newer text overrides older, but still name the old value. Explicit conflict marking fits when rank or time alone cannot decide: show 3000 and 5000 with dates and document types. For money, safety, or compliance without a single currently effective source, refuse or escalate instead of picking the number that reads more fluently.

**Deep dive:** After retrieval, align slots: from each chunk extract the “reimbursement cap” field plus number, date, genre, and status (pilot/revoked). Multiple values enter a referee: revoked marks drop out; an in-force policy beats casual mail; two policies compare effective dates; remaining ties emit a conflict object rather than a blended sentence. The generator must see the conflict flag; the prompt must forbid averaging, median, or “taken together” over numbers. Edge case: a new email claims “approved to 5000” while the policy page is unchanged—this is governance, not a retrieval miss. Mark “policy not updated; unapplied revision exists” instead of silently trusting the email. Detection can be rules (same field, different numbers) plus clustering; you do not need an LLM referee first. Measure conflict-detection rate and keep invented midpoints at zero.

**Follow-ups:** Who maintains the authority table, and what if a business owner reorders the wiki themselves? If the user says “follow last year’s email,” do you still place the policy beside it? If conflict detection sits after rerank and before generation, which is costlier—a miss or a false alarm?

**Extended questions:**
- Three sources say 3000/5000/3000. When is majority vote a valid referee, and when is it not?
- How do citations still show the rejected source so the user does not think the email was never retrieved?
- How do cases that must emit a conflict object stop the model from merging two numbers again?

**Reasoning:** A field value is either traceable to one effective source or explicitly marked as a conflict. “No invented midpoints” forces slot extraction then a referee, with money/compliance defaulting to refusal. Accept with conflict-detection rate and a zero count of midpoint values.

## RAG-035. A contract must be retrievable only by that customer’s owners. Why can’t you retrieve globally and then tell the model to keep it secret, and where in the ANN path should the permission predicate run?

**Answer:** Once a contract chunk is in service memory, the prompt, or logs, the leak has already happened; “do not output this” is a generation constraint, not access control. Permissions must be a retrieval predicate: narrow the graph or shard by tenant/customer first, then run ANN on that subgraph so forbidden chunks are never fetched. Post-filters are a last belt, not the only gate—they are slow at low selectivity, and any forgotten path (cache, debug UI, prefetch) leaks. Cache keys include the principal id; revocation must make old chunks unretrievable inside a stated SLA.

**Deep dive:** Store ACL on each chunk: customer id, roles, expanded principals, or a capability bitmap. The query carries the caller; pre-filter with shards or inverted bitmaps of allowed ids, then search HNSW/IVF; still re-check ACL when reading payloads so ids cannot be tampered. Do not rely on the model noticing sensitivity and staying quiet: prompt injection, citation rendering, and chain-of-thought logs will carry the source out. On membership change, update the visible set and invalidate that user’s cache as a group; revoke/delete needs stronger consistency than grant. Edge cases: parent expansion that stitches in a neighbor section without permission is a bypass; “similar contracts” search must run after pre-filter, or you will hit someone else’s contract, drop it, waste latency, and log their ids. Acceptance is an exfiltration suite: user A’s recall on user B’s documents is 0, and B’s document ids do not appear in the response.

**Follow-ups:** With five levels of nested groups, do you expand at query time or materialize members at ingest? Is revoke-from-allow SLA seconds or hours, and how does audit prove it? If search and generation are separate services, which layer is finally accountable for ACL?

**Extended questions:**
- If someone opens the vector files on a shared disk, do you still encrypt at rest or separate disks per tenant?
- When ops replay a production query, how do you stop them from seeing a contract they cannot access?
- If pre-filter makes one customer’s shard too small for stable ANN, how do you restore recall without lifting isolation?

**Reasoning:** Safety outranks recall: no invisible document’s text or id may enter the prompt or logs. That forces pre-filter/sharding as the main path and treats prompt secrecy as a soft extra. Accept with unauthorized recall = 0, cache isolation, and revoke SLA.

## RAG-036. One answer’s P95 is 3.2s: rewrite 200ms, ANN 80ms, cross-encoder rerank 900ms, generation 2.0s. Which segment do you cut, and what A/B keeps quality from collapsing?

**Answer:** Cut the largest segments that you can parallelize or shrink: stream generation to hide wait, shorten context to cut prefill on the 2.0s; on the 900ms rerank, A/B “50 candidates → 20,” a smaller reranker, or rerank only the first stage. ANN at 80ms is not this path’s bottleneck—do not start by swapping the vector store. Every latency cut must be scored on the same eval set for context precision@k and faithfulness, not seconds alone.

**Deep dive:** End-to-end is roughly the serial sum plus queueing. Rewrite and multi-query retrieval can often run in parallel; rerank and generation move quality. A cross-encoder scoring tens of chunks of a few hundred tokens each is easily a second at 50 candidates; 20 candidates or a distilled model often removes a large slice. Prefill grows with input tokens: dropping retrieved context from, say, 3K to 1.2K often beats a new ANN algorithm. Cache “normalized question + index version → reranked chunks” for head queries and the rewrite-to-rerank span can fall to milliseconds; invalidate on index change. Do not drop rerank to hit P95 until you have measured the quality delta with and without it, and only demote if the top-5 is already strong. Streaming changes perceived wait, not total time. Timers need a trace id and percentiles. This is not “ANN is slow on a huge corpus”: here the time is in rerank and generation.

**Follow-ups:** After 50→20 candidates, which question types fall out of recall; how much generation time remains on a prompt-cache hit; whether parallel multi-query doubles the rerank set and buys the latency back.

**Extended questions:**
- Draw the critical path and mark which stages can overlap and which must wait for rerank.
- If faithfulness is barely allowed to move on the eval set, how do you split the latency budget between rerank and generation?
- When do you rewrite with a small model and generate with a large one, instead of using the large model on both ends?

**Reasoning:** Sort by time, then attack rerank and generation. Run quality and latency on the same items. If ANN is already cheap, leave the index alone. Shorter context and versioned result caches are cheap levers; dropping rerank is an expensive one.

## RAG-037. On a million-vector corpus, P99 retrieval rose from about 80 ms to about 900 ms while the embedding model and reranker did not change. In what order do you inspect ANN parameters, filter timing, and caches before buying accelerator hardware?

**Answer:** If the model and reranker did not change, the cost sits after the query vector and before rerank: ANN walk, candidate filtering, and payload fetch. Ask first whether metadata filters run before or after graph search—post-filtering at a few-percent selectivity inflates candidates by tens of times and routinely walks P99 from tens of milliseconds toward a second. Then check whether HNSW `efSearch` or IVF `nprobe` was raised, whether inserts without optimize degraded the graph, and whether the index was paged out of RAM. Only then add caches for query embeddings and for `(query hash, filter)`. A GPU will not fix “search the whole corpus and throw away 95% of the hits.”

**Deep dive:** Split the timer: query embed (still ~10 ms if the model is unchanged) → ANN → metadata filter → payload → rerank (treat as a control). Post-filter cost is roughly `k_effective ≈ k / selectivity`: 5% selectivity needs about 20× over-retrieve, which blows up graph walk and random reads. Pre-filter or cut sub-graphs by tenant/time so each search lands on a million-scale piece, not the full set. During insert storms without a background optimize, HNSW long-range links rot and latency and recall worsen together. Cache keys must include the filter, or tenant A’s id list is served to B. Edge case: a hot-query cache repairs P50; novel tail queries still hit ANN, so P99 depends on sharding and filter timing, not cache hit rate. Hardware comes after residency, pre-filtering, and un-botched parameters.

**Follow-ups:** From one slow-query log, how do you tell graph walk from payload fetch? When selectivity swings with the user’s ACL, do you adapt `efSearch` or pre-partition? Can a cold page cache on a disk-backed index alone produce 900 ms?

**Extended questions:**
- If the filter changes from “tenant = X” to “tenant = X and tag ∈ {20 values},” how do you estimate post-filter over-retrieve?
- When the query-embedding cache hits but the document just changed, how much hit rate do you give up for freshness?
- How do staged timings and “candidates left after filter / k” prove the slowness is filter timing, not the ANN implementation?

**Reasoning:** The goal is P99 back in the low hundreds of milliseconds without collapsing recall. Exclude the unchanged model and reranker, estimate post-filter over-retrieve from selectivity, and only then touch hardware. Verify with staged latency, candidate inflation, and shard size.

## RAG-038. When indexing five million internal documents for RAG, how do you use order-of-magnitude estimates to choose sharding, storage, and incremental rebuilds instead of buying bigger machines first?

**Answer:** Convert document count into vector count and resident bytes first, then decide how large a graph still fits in memory per shard and how many embeddings a change must recompute. Five million documents at about eight chunks each is about forty million vectors; 768-d float32 raw vectors are roughly 115 GiB, and HNSW adjacency often adds another 50–100%, so a full in-memory index is often 180–230 GiB. That scale should not be one graph forced onto one box: shard by tenant or namespace into million-vector pieces, and re-embed only dirty documents via content hashes. Bigger machines buy RAM; they do not shorten a full re-embed or purge stale deletes.

**Deep dive:** The pipeline is parse → chunk → content hash → embed → write a sharded ANN; a query is routed only to shards the caller may see. Illustration: `n_vec = 5e6 * 8`, `bytes = n_vec * 768 * 4 ≈ 1.2e11`. IVF plus coarse quantization can drop codes to tens of bytes per vector, but recall then depends on `nlist` (often on the order of `√N`, so a few thousand to tens of thousands of lists at 40M) and `nprobe`; HNSW is steadier on latency and hungrier on RAM. Do not rebuild the corpus on every edit: skip unchanged hashes; write tombstones for deletes and drop them from every replica inside the SLA, or “ghost chunks” will occupy top-k for a long time. Edge cases: sharding only by time misses cross-year policy questions; sharding only by tenant still needs a second cut if one tenant already has tens of millions of chunks. Re-embedding 40M chunks at about 1k chunks/s is hours, so “buy RAM and rebuild everything” makes the change window unacceptable.

**Follow-ups:** Do you size a shard by memory, QPS, or recall variance? While a tombstone has not replicated, should the query fail, degrade, or read the old chunk? After PQ, how many recall points will you accept, and on which frozen eval set?

**Extended questions:**
- If 5% of documents change every hour and the rest barely move, how do dirty-set re-embed and a nightly full rebuild differ in CPU/GPU budget?
- After the shard key becomes “tenant + document type,” how many shards does a cross-type multi-hop question hit?
- How do “index bytes / vector count” and “chunks still retrieved after delete” tell you the capacity plan was wrong?

**Reasoning:** Constraints are RAM residency, change window, and delete visibility; the goal is recall that does not collapse with scale. Map documents to vectors and GiB to see whether one graph fits; compare “hash-incremental + shard routing” with “vertical scale then full rebuild.” Accept with per-shard recall@k, index lag, and ghost-chunk rate—not with whether the machines look big enough.

## RAG-039. The same policy lives in a wiki, meeting notes, and a PDF, so top-k is filled with one conclusion. Which evidence do document-id, adjacent-window, and semantic dedup each destroy by mistake?

**Answer:** Dedup by document id drops two relevant sections of the same file (definition and exception) and does nothing to cross-source copies. Adjacent-window dedup eats overlap clones and misses a wiki paraphrase of a PDF. Semantic dedup (for example keep one when cosine is above ~0.9) compresses cross-source copies but can drop a minority source that is phrased alike and flips the number. Layer it: canonicalize and hash true duplicates at ingest; merge same-document intervals at query time; at semantic clustering, keep every member if a critical slot (amount, date, yes/no) disagrees.

**Deep dive:** The three copies usually differ—short fresh wiki, spoken notes, layout-noisy PDF—yet still sit close in embedding space, so top-k looks full and entropy is low. At ingest, normalize URLs, simhash title+body, and record attachment derivation so aliases point at one canonical id; retrieve the canonical by default and expand aliases for audit. MMR increases vector diversity but does not test slot conflicts, so it is not a substitute for clustering then comparing numbers. Edge case: one file says “pilot,” another “revoked,” and they are semantically close—keeping only the top score systematically deletes the revocation. Measure “unique claims / k” and whether conflicting slots are kept together; do not KPI on “rows removed,” which can be accidental deletion.

**Follow-ups:** Is the canonical copy the wiki or the signed PDF? Do you set semantic thresholds per document type so short titles are not marked duplicates? If dedup runs before versus after rerank, what happens to cross-encoder scores?

**Extended questions:**
- Three copies of the same claim span two years. After dedup, which one do you keep, and does rank score or freshness win?
- Near-duplicate detection with character n-grams versus vectors: which one does noisy OCR break first?
- How do you build an eval set of “many agreeing copies plus one counterexample” so semantic dedup cannot silently drop the counterexample?

**Reasoning:** The goal is different evidence in top-k, not different filenames. Choose the dedup key by the damage it does—identity, geometry, semantics—and add slot-conflict protection at the semantic layer. Accept with unique-claim count and counterexample retention.

## RAG-040. After you set chunk overlap to about 20%, five of the top-8 hits are sliding windows of the same paragraph. What do you give up if you lower overlap at ingest versus merging by offsets after retrieval?

**Answer:** The 20% overlap is there so sentences are not cut, but when neighbors share about a hundred tokens their vectors are near-duplicates and ANN fills top-k with clones. Dropping ingest overlap to about 5–10%, or cutting on headings with a one-sentence overlap, removes clones at the source and makes anaphora and cross-sentence definitions easier to split. Merging intervals on `(doc, offset)` after retrieval (for example union when IoU exceeds ~0.3, then expand once) frees slots immediately without re-ingesting, but a jittery score may drop the only window that held the full sentence. They are not substitutes: keep cuts from breaking key sentences, then merge geometric overlaps at query time.

**Deep dive:** A 512-token chunk with 102-token overlap often has adjacent cosine above 0.85, so five sibling windows in top-8 is unsurprising. Interval merge needs stored `start/end`, grouping by document, keeping the highest-scoring window or expanding the union once so five hits do not become one starved snippet. MMR picks embedding diversity; two geometrically overlapping windows with slightly different wording can both survive, so MMR is not a sliding-window merger. Edge case: a table row split across chunks gets worse if you simply reduce overlap—cut on structure instead of raising overlap again. Watch context precision and “unique parent passages in top-k,” not recall@k alone; clones inflate recall.

**Follow-ups:** After a merge, do you re-embed the union and rescore? For heading-less notes, do you cut on sentence boundaries or token windows? After overlap falls from 20% to 5%, which query class loses recall first?

**Extended questions:**
- Three sliding windows of the same paragraph score 0.84/0.83/0.82. How does the merge rule avoid always keeping the middle window that lacks a subject?
- If you both lower ingest overlap and merge hard at query time, how do you stop an exception sentence from disappearing?
- How would “share of top-8 that are sibling windows” become a ship gate so you do not wait for users to call the answer repetitive?

**Reasoning:** Top-k slots are scarce; the goal is one slot per claim without cutting sentences. Compare the queries each tactic harms—“less overlap” versus “offset merge”—and stack them. Accept with unique-parent count and misses caused by broken sentences.

## RAG-041. Support collected about 200 downvoted answers in a week. How do you cheaply label six failure types, then let the mix decide whether to fix crawl, chunking, or generation checks first?

**Answer:** Treat the 200 items as a failure-mix sample, not 200 single-ticket teardowns. Use six mutually exclusive primary causes: missing document, chunking split, ranking miss, stale version, generation ignoring context, and wrong question interpretation. Label them with six cheap probes—index membership, span-in-chunk, chunk IDs in the prompt, version hash, whether the gold sentence is already in the model input, and whether the rewrite dropped a constraint—and send only the residual to humans. Invest in the layer that is about one-fifth of the mix, or a smaller slice that is all money or ACL errors; do not start six parallel rewrites.

**Deep dive:** Two hundred examples can separate a 15–20 point share gap and cannot separate a 2-point one, so the job is to pick the next iteration, not to staff six workstreams. Each row must replay against the index revision and the full model input from that request; otherwise “it is in the corpus now” mislabels a miss or a stale hit. Cascade probes in earliest-break order and stop at the first hit: (1) the document, policy name, SKU, or error code named by the user or a reviewer is absent from that snapshot → missing document; (2) the document is present, but the gold sentence or number is split across chunks or lives only in the raw parse → chunking split; (3) the gold chunk exists but is missing from the logged prompt set (present in top-50 before rerank, gone after rerank, still counts) → ranking miss; (4) the hit’s content hash or effective date is not the source system’s then-current version, or it is marked withdrawn → stale version; (5) the gold sentence is already in the model input, yet the answer’s number, polarity, or object disagrees → generation ignoring context; (6) the rewrite dropped a negation, object, or time window and retrieved a coherent but different question → wrong question interpretation. The first five are point lookups, regex over the chunk table, set ops, hash compares, and substring checks on the prompt—milliseconds to tens of milliseconds each. The sixth class plus leftovers (for example about 30–40 rows) get one to two minutes of human time. One ticket gets one primary label—“fixing this layer should have been enough”—with optional secondary tags so a single miss is not counted six times. Spend against weighted counts: if about 80 rows are missing documents, fix connectors and coverage before swapping embeddings; if generation-ignoring-context leads, add citation constraints or claim checks, do not rechunk the whole corpus. After a fix, relabel the same 200 plus the next week of downvotes; the mix will move. Two hundred rows cannot power six parallel A/Bs; they can lock one primary iteration.

**Follow-ups:** With no user-pasted gold document, how do you use an entity dictionary to approximate “missing document” without tagging rare-but-indexed questions as gaps; if a ticket is both split and freely generated, do you primary-label chunking or generation, and how does the regression avoid turning green after fixing only one layer; how do you drop “factually right, tone wrong” downvotes so they do not warp the mix.

**Extended questions:**
- What minimum log fields does each probe need (index revision, chunk_id list, prompt text, source hash, query before and after rewrite), and which missing field collides two classes?
- Stale version is only about a dozen rows but all amounts; missing document is sixty rows on fringe products. Do you still follow raw counts or risk-weight the next iteration?
- How do you validate the taxonomy by asking “if we had fixed this layer, how many of the 200 would flip to correct,” instead of declaring chunking most important by instinct?

**Reasoning:** The goal is to spend the next iteration on the true modal—or highest-risk—failure. The constraint is that 200 tickets cannot pay for a full human trace each, and cannot pay for turning six knobs at once. So machine-runnable layer probes become a mutually exclusive primary cause, share plus risk-weight ranks the work, and you fix one class then remeasure whether the mix actually moved.

## RAG-042. End-to-end accuracy is flat, faithfulness fell, and context recall is still high. Do you debug retrieval or generation first?

**Answer:** High context recall means the points needed to answer mostly made it into the prompt. Falling faithfulness means the answer now contains claims the supplied context does not support. That is unfaithful generation, not a miss. Freeze retrieval and inspect prompt constraints, noisy neighbors, and output checks. Only reopen the retriever if context precision collapsed as well, or the gold document is not in the top-k at all.

**Deep dive:** Four metrics watch four stages. Context recall: of the annotated points required to answer, how many appear in this turn’s retrieved context. Context precision: of the retrieved chunks, how many are actually useful, and whether the useful ones are ranked high. Faithfulness: atomic claims in the answer, each entailed by the given context. Answer relevance: whether the reply addresses the user’s question rather than being correct but off-topic. They fail independently. The right paragraph can arrive together with several similar old policies; the model then invents a number that is in none of them—recall stays high, precision dips, faithfulness drops. If a model is the judge, lock the context boundary: it may use only this turn’s retrieval, not parametric fill-in. Each labeled item should carry the question, the retrieved list, the answer, the necessary points, and planted distractors. Thumbs-up online is not claim-level review; users will accept something that “sounds right” after faithfulness is already gone. Spot-check by splitting each answer into 3–8 claims rather than a single right/wrong bit.

**Follow-ups:** How you stop a model judge from inventing support; whether context precision should be binary relevance or rank-discounted; whether abstentions get their own faithfulness bucket.

**Extended questions:**
- For each of the four metrics falling alone, which pipeline stage do you change?
- How do you build two adversarial sets: “retrieval right, answer wrong” and “retrieval wrong, answer accidentally right”?
- When citation accuracy and faithfulness disagree, which one gates a release?

**Reasoning:** Map metrics to stages: recall/precision are retrieval; faithfulness/relevance are generation. High recall plus low faithfulness means the generator is speaking beyond the evidence. Add “answer only from context or abstain” and claim-level checks before you touch the retriever.
