# Model Internals: Transformers and Language Models

[← Back to contents](../README.md)

## ARCH-001. Ticket routing is an 8-way closed set, while copywriting must produce paragraphs that never appeared verbatim. Why is a discriminative model usually safer for the first and a generative model required for the second?

**Answer:** A discriminative model learns P(label | input) and emits a closed set, so calibration and abstention are straightforward. A generative model learns a distribution and samples new artifacts that resemble the training data without being copies, which is how it composes unseen sentences. Routing answers live in a closed set; open generation adds format drift and invented labels. Copy lives in an open token space; a classifier cannot emit a paragraph.

**Deep dive:** Text is usually factored autoregressively (next-token); images often use iterative denoising, with older GAN/VAE lines still in the family. Generation is sampling, so one prompt can yield many strings; you must handle fluent-but-false hallucinations, controllability, copyright, and safety. Discriminative failures are boundary mistakes and miscalibration. Generative failures are fabricated facts, runaway length, and policy-breaking content. Production systems often split the job: classify or gate with a closed model, write with a generator. Metrics do not transfer either—accuracy/AUC suffice for routing; generation also needs factuality, diversity, and human spot checks.

**Follow-ups:** Why a generator can fluently invent an order id that never existed; if you still classify with a generator, how temperature and constrained decoding reduce format risk; when “ask the generator for a label” is still reasonable.

**Extended questions:**
- Contrast autoregressive and diffusion generation on parallelism, length control, and mid-sequence editability.
- When the same prompt disagrees across calls, do you change sampling, cache, or switch to a discriminative head?
- Name production tasks that must not be open generation and should stay rules or classifiers.

**Reasoning:** Inspect the output space: closed labels versus open sequences. Closed → discriminative; open content → generative. Then price the failure: a wrong bucket can be retried; a fabricated clause is expensive, so the generative path needs rails and an eval set.

## ARCH-002. When you stop training one small model per product line and share a single foundation model, what disappears from the engineering ledger and what shows up instead?

**Answer:** What disappears is the main path of collecting labels, training weights, and babysitting a training cluster for every classification or extraction task. What shows up is a product system around one general model: prompt and context assembly, retrieval and tools, eval sets, safety filters, routing, and token billing. A foundation model moves the capability ceiling from “what classifier this labeled set can support” to “what generation this context can constrain,” but the model is not the product—authorization, refusal, and fallback still belong to the business. Typical signals that a line should not be forced onto the shared model: calibrated probabilities, hard on-device latency, or outputs that must be auditably deterministic.

**Deep dive:** A foundation model is pretrained with a self-supervised objective on a huge mixed corpus (commonly next-token prediction), yielding transferable representations and a generation prior, then adapted with prompts, RAG, tools, or light fine-tuning. The engineering center of gravity shifts from “feature work + one checkpoint per task” to “context budget + version drift + hallucination or injection incidents.” The cost structure flips with it: the main bill is often inference tokens × unit price, not GPU-hours from retraining twenty small models each month. Eval headcount does not vanish; it expands from offline accuracy to traces (prompt, retrieved spans, tool I/O, output). Failure modes differ too: a small classifier emits a wrong closed-set label out of distribution; a foundation model fluently invents a paragraph that looks like policy. Observability therefore has to replay a full request, not just a task-score dashboard.

**Follow-ups:** Under what data volume and latency constraints are prompting, retrieval, and SFT each worth it; how do you regress after a vendor model revision so quality does not silently drop; which business metrics show that total cost of ownership fell, rather than GPU utilization alone.

**Extended questions:**
- For a closed-label QA task with only a few thousand stable labels, why might you keep a small classifier?
- After the foundation model is wired in, should the permission boundary sit in the prompt layer, the retrieval layer, or the tool sandbox?
- How would you compare the old small model vs the foundation model + RAG on shadow traffic so you are not judging from one pretty reply?

**Reasoning:** Start from the goal “reuse one general prior, stop training many weight sets” and the constraints of correctness, authorization, and rollback. List the training pipelines that go away and the inference-time control plane that appears, then keep a specialist model where the new failure shape is unacceptable. Validate three things: task scores do not drop, a request is replayable, and token cost can be attributed per product line.

## ARCH-003. Records must stay on the intranet and you serve 20 million tokens a week. When should you refuse an external closed API?

**Answer:** Start with residency: if notes, prompts, or logs cannot leave the campus, an external closed API is out, no matter how strong the model is. If compliance would allow it, run the numbers—20 million tokens a week rarely pays for a dedicated-GPU self-host (power, idle time, on-call), so a closed API is often cheaper. Refusing the API takes three things at once: a hard compliance line, an acceptable quality gap, and a team that can evaluate, patch, and page outages. The usual landing spot is routing: requests that carry records stay on an on-prem open-weight model; only a few de-identified hard cases may go to a controlled external model.

**Deep dive:** “Open weights” means you can obtain parameters and the tokenizer. It does not mean the license allows commercial use, and it does not mean you can audit alignment data or safety policy. A closed API buys ongoing upgrades, tooling, and a contractual SLA; you pay with non-reproducible outputs, rate limits and prices you do not control, and contract language about whether prompts may be used to improve the service. Order-of-magnitude: 2×10⁷ tokens/week is about 10⁹ tokens/year; at publicly quoted “a few dollars per million tokens,” yearly model spend often lands in the hundreds to low thousands of dollars, below depreciation on a datacenter GPU. If you need first token <200 ms on campus, offline fine-tuning, or turn-by-turn audit replay, control is worth more than the sticker price. Edge cases: being able to download weights is not permission to ship; commercial clauses, output watermarking, and a missing ops bench can all kill an “open” plan. Score quality evals, p95 latency, cost per token, residency, fine-tune rights, and exit lock-in side by side—not a public chat leaderboard alone.

**Follow-ups:** Which legal risk sits on open weights versus open training data? If a vendor offers a dedicated private cloud, is that still the closed API you wanted to refuse, and do you still self-host? If quality is a few eval points worse, can retrieval and tools close the gap?

**Extended questions:**
- At 2 billion tokens a week, how do you estimate the cost crossover between self-hosting 7B / 70B and staying on an API?
- How would you build a notes eval set that shows an on-prem 14B can replace an external large model on this task?
- If a router sends a field that was not de-identified off-campus, which request metadata should monitoring capture?

**Reasoning:** Order constraints as residency/compliance → minimum task quality → latency and fine-tune needs → cost. Use the red line to drop APIs that cannot stay on campus, then estimate TCO from token volume so you do not self-host “for autonomy” while volume is still too small. Accept with sliced evals, refusal rate, latency, and audit logs—not brand reputation.

## ARCH-004. A 4k-token training example can compute loss in one parallel pass, yet producing 200 tokens still takes 200 sequential steps. Why does the same autoregressive factorization yield two completely different degrees of parallelism?

**Answer:** Autoregression factors the joint as ∏_t P(x_t | x_<t). At train time the true prefix is already in the example; a causal mask only blocks looking ahead, so every position can predict the next token in one forward pass and one cross-entropy. At generation time the next input is the token just sampled. That is a real data dependence a mask cannot remove, so decode steps grow linearly with output length.

**Deep dive:** Training is dominated by a [batch, T, d] multiply laid out across T. Decode effectively has sequence length 1, so utilization drops. A KV cache keeps already computed K and V so step t does not reproject the first t−1 tokens: compute falls from O(t) to “new position only,” but cache reads still grow with t and long outputs often become bandwidth-bound. Time-to-first-token is the prompt prefill; later token latency is one decode step plus cache traffic. A bad sample at step k enters every later condition and errors accumulate down the chain. Speculative decoding and multi-token prediction cut serial steps without changing the chain rule; parallel denoising models replace the factorization itself.

**Follow-ups:** Why does a causal mask unlock training parallelism but not generation? Which part of the data path is teacher forcing, and which is exposure bias? Does the KV cache save compute, memory, or swap one for the other? If the training target predicts the next 4 tokens at once, how do serial step count and calibration error change?

**Extended questions:**
- With a 2k prefill and 512-token decode, roughly what share of end-to-end latency is serial steps, and how does latency jump if the cache is evicted?
- Does the same serial bottleneck appear when you autoregress over image patches or audio frames? What tricks advance several steps at once?
- If training computes loss on prefixes the model itself just produced (no teacher forcing), how do throughput and early collapse risk change?

**Reasoning:** In the chain rule, x_<t is a dataset constant during training and a random output of the previous step during generation. Constants can be batched across T; random outputs draw sequential edges in the graph and drop parallelism from T to 1. Speedups either cache intermediates, predict several future positions, or abandon the factorization.

## ARCH-005. A support reply reads like a human agent but cites the wrong policy — is that corrupted weights, or is that how generation works?

**Answer:** It is the mechanism. A large language model tokenizes the input, a Transformer produces a next-token distribution over the vocabulary, and greedy or sampled decoding writes a continuation. The training objective makes the continuation sound like support-desk text in the mixture, not look up a guaranteed-correct policy table. Fluent but wrong policy means a high-likelihood “support voice” was sampled; factuality was never in the loss. Corrupted weights more often show up as NaNs, garbage, or a crash, not polished mix-ups. Put constraints on retrieval (inject the current policy), decoding (refuse or force a citation when unsupported), and post-processing (rules or a tool check)—do not start by reloading the checkpoint.

**Deep dive:** The data path is tokenize → embedding → L layers of causal self-attention and FFN → LM head logits of shape about `[vocab]` → temperature / top-k / top-p sampling. Vocabularies commonly sit from tens of thousands to a few hundred thousand; there is no separate “truth head.” Temperature only sharpens or flattens the distribution; it does not turn a wrong clause into the right one. When the context lacks the policy, or retrieval returns a stale or competitor snippet, the model fills the hole with similar phrasing from pretraining—and fluency can even go up. Unlike a classifier, there is no closed label set, so a string that merely looks like an answer is a legal sample. Online, log “did we cite a policy span, and does that ID hit the knowledge base,” not just BLEU or a human impression of polish.

**Follow-ups:** For the same hallucination, how do raising temperature vs turning RAG off each change the rate; if policy updates weekly, which path has shorter lag—weight fine-tuning or retrieval refresh; how a gold set of “refuse unless grounded” items separates model skill from assembly bugs.

**Extended questions:**
- If decoding may only copy spans from the policy text, which conversational abilities do you give up?
- In a multi-turn thread, stale policy left in history—does the model trust history or the new retrieval?
- How would you replace “the model recites policy from memory” with a tool call to a policy API?

**Reasoning:** Align the training objective (next-token likelihood) with the inference interface (a conditional language model), then classify the incident as “fluent and ungrounded” vs “numerics blew up.” Fix the first as missing system constraints; only the second is a checkpoint issue. Accept on policy-check pass rate and refusal rate, not on fluency.

## ARCH-006. The same mixed Chinese–English code snippet is ~800 tokens on one model and ~1400 on another. How do you tell product this is not “creative billing”?

**Answer:** Token count is defined by that model’s vocabulary and segmentation rules, not by characters or words. Mixed language, identifiers, whitespace, and punctuation merge very differently across vocabs: an English-heavy table often falls back to shorter pieces or bytes on CJK and symbols, so the same ~100 lines can move from about 800 to about 1400 (always measure with the real tokenizer). B is not necessarily overcharging, but the same user-visible string consumes more of B’s context window. Truncation and invoices must use the serving tokenizer, not a character budget. What you align with product is “the token bill of this request under this model,” together with unit price and quality.

**Deep dive:** A rough English rule of thumb is sometimes “about 4 characters per token”; Chinese on an English-heavy vocab often lands near one character per token or worse. CamelCase, indentation, and operators fragment further. Role separators and BOS/EOS add a fixed per-turn overhead. The window simultaneously holds system prompt, history, retrieval, and reserved output; 800 vs 1400 means ~600 fewer tokens for retrieval and multi-turn history—the window hits the wall before anyone’s intuitive “word count.” Billing is typically (input tokens + output tokens) × price; if you audit B’s invoice with A’s counts, it will never match. Boundary: byte fallback can turn one Unicode character into several tokens; a domain word that made the vocab collapses to one. Before launch, histogram p50/p95 tokens on production samples, not an English-prose coefficient.

**Follow-ups:** How you explain to finance that the same feature on two models can honestly differ by 2× tokens; what breaks if you truncate on characters vs tokens in each language; which part of the input bill usually contains special tokens and the tool schema.

**Extended questions:**
- Why can you not feed model A’s token-id sequence straight into model B?
- If Chinese token efficiency is an acceptance gate, which real tickets should you sample instead of news text?
- How does squeezing the prompt (stripping spaces, abbreviating code) save tokens while hurting the model’s ability to read the code?

**Reasoning:** Start from “the billing unit is the length of this model’s tokenizer output,” drop the character-alignment illusion, and explain the price and truncation gap with actual encode counts and remaining window on the same string. Accept with a reproducible encode script and a p95 histogram, not a verbal conversion.

## ARCH-007. For a Chinese–Japanese–English support desk, do you whitespace-split then WordPiece, or feed raw strings to SentencePiece? Decide with two examples that break.

**Answer:** Prefer SentencePiece on the raw unsegmented string (BPE or Unigram). Chinese and Japanese usually have no spaces between words, so a whitespace pre-tokenizer treats a whole Chinese sentence as one “superword,” and WordPiece can only smash it or blow the length. WordPiece-after-whitespace typically breaks on `退货政策如下：` as one block, or on English `don't` / `SKU-123` if punctuation rules disagree with training. SentencePiece typically breaks when the corpus is Chinese-heavy: English code and URLs shatter and p95 tokens spike. I would pick SentencePiece and roll back if p95 tokens and junk splits on English/code tickets exceed budget—then retrain or add a code-aware pre-split.

**Deep dive:** WordPiece also learns subwords, but merges lean toward improving LM likelihood; encoder stacks in the BERT family often use it, mark continuations with `##`, and historically assume a whitespace/punctuation word pre-tokenizer. SentencePiece treats space as a normal character (often shown as `▁`) and does not need a language-specific segmenter, which fits CJK. Unigram pruning starts from a large vocab and drops pieces to maximize likelihood instead of greedy merges. Either way, serving must use the exact same vocab and preprocessor as training—swapping the tokenizer is swapping the model. Two gold failures are enough: a space-less Japanese address (scheme A tends to swallow or shred the block) and an underscored English order id (scheme B on a Chinese-majority vocab may byte-split). Lock those two encode sequences and the detokenized strings; do not argue from “friendlier to Chinese.”

**Follow-ups:** If traffic is 90% Chinese and 10% code, do you route two tokenizers; what boundary does WordPiece’s `##` vs SentencePiece’s `▁` store in the embedding; after you retrain a tokenizer, how must old SFT corpora be re-encoded.

**Extended questions:**
- With zero-width characters or full-width spaces, which pipeline is bypassed first by a safety keyword check?
- If keys inside tool JSON are over-split, how does that raise malformed tool-call rates?
- Why does “generic segmenter, then map into the model vocab” almost always lose quality?

**Reasoning:** Constraints are CJK without spaces plus stable English/order-id splits, so drop anything that requires whitespace pre-tokenization, then hold SentencePiece’s English regression with p95 tokens and gold round-trips. Accept on fixed-sample encode/decode and per-locale percentiles, not language intuition.

## ARCH-008. After pretraining switches from next-token to 15% random masking, why is the model better as a sentence encoder and almost unable to stream chat?

**Answer:** Random masking lets each hidden position see left and right context, so the model learns bidirectional fill-in-the-blank; pooled sentence vectors fit symmetric similarity. Chat needs the causal factorization p(x_t | x_<t) and the ability to emit one token at a time. A masked objective has no such chain; forced generation is “pick a site, fill the mask, remask,” which cannot stream. Changing the pretraining target changes the attention mask and the output head. It is not a coat of paint.

**Deep dive:** Autoregressive: input [t₀ … t_{n−1}], target [t₁ … t_n], causal mask, loss at every position, dense supervision per forward. Masked language modeling: typically about 15% of tokens become mask / random / original, loss only there, each prediction uses full bidirectional context. The inference loop is different: AR uses a KV cache with O(1) projection and O(t) attention per step. A masked model that generates must pick a length and iteratively rewrite undecided sites; step count is tied to quality, and you cannot push the first token to the user as it appears. Magnitudes: the same 8B model on a 4K sequence has 4096 supervised positions under AR and about 614 under 15% masking—sparser gradients, but each signal carries full bidirectional context. Edge cases: span corruption and prefix LMs (bidirectional prefix, causal suffix) sit in between. A common production split is a masked/bidirectional model for retrieval and an AR model for generation. “Can fill a hole in a sentence” is not “can chat.”

**Follow-ups:** Why can’t you casually raise 15% to 50%? For AR sentence vectors, which is weaker—last-token or mean pooling—and how do you fix it? Where does latency differ between non-AR iterative decoding and a real decode loop?

**Extended questions:**
- If you use a 15% masked model for translation, what extra structure do you need before it can decode autoregressively?
- On the same corpus, which objective usually burns more tokens to reach the same validation loss?
- What happens in the forward pass if a chat prompt has a mask token inserted in the middle?

**Reasoning:** Back out the probability factorization the product needs: streaming chat wants causal conditionals; sentence vectors want bidirectional interaction. Align those with each objective’s mask, loss sites, and inference loop, and “better vectors, no chat” follows. Accept with retrieval metrics and a yes/no on streaming tokens.

## ARCH-009. Someone wants to replace your online chat model’s Transformer backbone with an LSTM to “save memory.” Which compute arguments do you answer with, and when do you admit a recurrent net is still the better fit?

**Answer:** At training time a Transformer can GEMM over sequence positions in parallel under a causal mask; an LSTM is serial in time, so wall-clock on a huge token corpus is usually unacceptable. For long-range dependencies, attention lets position i read j in one hop; an LSTM must carry the signal through O(distance) hidden-state steps and tends to drop distant constraints. At decode, the “save memory” claim is partly fair: an LSTM stores O(d) state per layer, while a Transformer KV cache is O(sequence length × d) per layer. Modern decode still uses that cache to avoid recomputing history—it is not an extra recurrence versus LSTM, just a larger cache. Recurrent nets still win when the device has only tens of MB, sequences are tiny control traces, or you explicitly trade away long-range modeling for constant-size state.

**Deep dive:** A typical decoder-only block is embedding (plus some position injection) → L times (norm → multi-head self-attention → residual → norm → FFN → residual) → final norm → LM head. The pairwise attention term grows about O(n²d); projections and the FFN grow about O(nd²). For an illustrative ~7B model with width around 4096 and ~32 layers, a 4k training context fills a GPU with large GEMMs; an LSTM of the same width serializes the time steps and utilization collapses. At inference, once context reaches thousands of tokens, KV is the memory hog—cut it with GQA, quantization, or windowed attention rather than reverting to LSTM. LSTM saves KV but cannot pay the pretraining cost at the same scale and cannot reuse the current parallel kernels. Boundary: length 1–2 and a hard constant-size state, a recurrent layer as a light controller is still reasonable.

**Follow-ups:** At the same ~7B scale, which side wins on training throughput vs decode KV memory, written as a one-page estimate; which complexity term linear attention or state-space models cut relative to LSTM/Transformer; on-device, how you pick distilled small Transformer vs LSTM from a latency–quality curve.

**Extended questions:**
- If you only replace the FFN with a recurrent mix and keep attention, do you still get parallel training?
- After INT8 KV quantization, what order-of-magnitude gap remains versus an LSTM’s state volume?
- Causal Transformer decode is serial too—does the training-time parallelism argument still hold?

**Reasoning:** Split “save memory” into peak training memory, decode KV, and wall-clock throughput, and score each with parallelism and complexity so you do not let decode KV volume argue against why training needed a Transformer. Re-evaluate on-device under a hard RAM cap. Use reproducible throughput and memory logs, not architecture taste.

## ARCH-010. You want a vocab of about 30k. Why does BPE’s merge order pack a frequent English word into a couple of pieces while a formula that appears three times is shredded?

**Answer:** BPE starts from characters or bytes and repeatedly merges the most frequent adjacent pair in the current corpus into the vocab until you hit about 30k (or whatever cap you set). A word that appears a hundred thousand times has internal pairs that merge early and often ends as 1–2 subwords. A formula that appears three times never outranks mass pairs like “th” or “ing,” so it stays shredded. After a domain shift, inspect term fertility (tokens per word) first: a smashed term wastes window and never becomes a stable unit.

**Deep dive:** Train: count pair frequencies → merge the winner → refresh adjacency → repeat. Encode by greedily applying the learned merges left to right (library details vary; the merge table is the contract). For example, if `unhappiness` is globally frequent you may first get `ness` and `happy`, then longer chunks; `C6H12O6` is a rare digit/letter interleave with single-digit pair counts, so the vocab often keeps only character-level shards. Byte-level BPE avoids UNKs, but rare Unicode and formulas can still split on bytes—one visible character, several tokens. A larger vocab can “remember” mid-frequency domain terms, at the cost of a bigger embedding matrix (about `vocab × d`). Boundary: BPE does not owe you morphology; `un-happiness` and `unhappi-ness` are both legal and depend on corpus counts, not a grammar book.

**Follow-ups:** If the merge rule maximizes language-model likelihood (WordPiece-style), do rare terms fare better; growing the vocab toward 100k, what do you trade in embedding memory vs Chinese/code fertility; how domain-term coverage tells you whether to retrain the tokenizer on vertical text.

**Extended questions:**
- Why can a finished BPE table not keep merging online on new traffic at inference?
- When numbers are split to single digits, what error pattern shows up in addition and subtraction?
- If fertility on English prose vs JSON logs differs by an order of magnitude on the same vocab, whose budget should set the context cap?

**Reasoning:** The goal is a fixed vocab that compresses frequent pieces and avoids UNK, so merges reward frequency and rare formulas must fragment. Accept with fertility and unknown-byte rates on a domain term list, not whether a few common English words encoded as whole tokens.

## ARCH-011. Drug names shatter into six or seven subwords, a batch of terms fills half the window, and classification scores drop. Can a finished model take new vocab rows as-is, or do you quantify fertility first and then choose a fix?

**Answer:** First measure term fertility (mean subwords per term), the share of input tokens they consume, and whether the task recovers if you artificially merge the same terms into a shorter sequence — confirm it is a segmentation problem, not label noise. A finished model cannot only swap the tokenizer file: new tokens have no aligned embedding or LM head. You initialize them (for example the mean of their subword vectors) and continue pretraining, or at least train the embeddings. A from-scratch domain vocabulary segments cleanest and costs most; byte-level has no OOV but sequences can still be long. Without touching weights, normalize standard names to an already-short token sequence at inference, or put a glossary in retrieval.

**Deep dive:** Chinese drug names, formulas, and product SKUs often have no whole-word piece in a general BPE/Unigram. At fertility 6–8, 128 terms can consume nearly a thousand tokens; an 8k window loses real sentences and the classifier sees fragment co-occurrence instead of entities. Growing the vocab without training leaves new rows as random or zero vectors, out of distribution, and generation treats those ids as noise. Mean-of-pieces init makes the first step look like “one word,” but you still need enough steps for attention to use it as an atom. Vocab growth costs softmax and embedding memory linearly; doubling the table for a few hundred drug names is usually a bad deal — add a small domain block and continue training. Lightweight normalization (alias → generic name, strip spaces, unify trade names) often precedes any vocab change. Accept on term fertility, window occupancy, and entity-level F1 together, not on whole-sentence accuracy alone.

**Follow-ups:** For a new token’s embedding, is mean-of-pieces, zeros, or a copy of a near synonym more stable in the first continued-training steps? If you grow the table but leave the segmentation rule unchanged, can old text be cut in a form that coexists with the new id? Should Chinese, Latin, and trade names of one drug become one token or three?

**Extended questions:**
- Write a stats script: term list plus tokenizer in, mean fertility and share of all tokens out.
- If you may only continue-train embeddings and freeze the rest, which tasks recover, and which (e.g. generating a drug name) stay shattered?
- When mixing a domain vocab with a general vocab, how do you avoid merging ordinary Chinese words into ambiguous pieces?

**Reasoning:** The goal is less term occupancy and recoverable entities; the constraint is that existing embedding alignment cannot be casually broken. Use fertility and occupancy to prove segmentation is the root cause, then “changing segmentation implies changing and aligning weights” rules out “edit the tokenizer file only.” Pick among normalize / add a small vocab and continue-train / retrain the tokenizer by budget. Validate that term token count and entity score move together, not that the vocab got larger.

## ARCH-012. Vector search ranks “returns allowed” next to “returns not allowed.” Is the similarity formula wrong, or was the embedding never asked to separate negation?

**Answer:** Usually the training objective, not a broken cosine. Contrastive sentence embeddings pull paraphrases and topically related text together; the two refund sentences share almost every word and scene and differ by one negation, so they can sit close in geometry. Cosine only measures an angle in that space; it does not implement entailment. Switching to dot product or L2 rarely splits negation pairs. Hybrid retrieval, a negation-aware reranker, and hard filters on product or policy version work better. Also do not treat the RAG sentence vector and the LLM’s input token-embedding table as the same object.

**Deep dive:** The LLM’s token embedding is a `[vocab, D]` lookup that seeds the first layer. RAG sentence vectors usually come from a dual-encoder or instruction-tuned encoder with pooling. That encoder is trained so queries sit near positive docs and away from negatives; negatives are rarely mined as logical opposites, so antonyms become false neighbors. If chunking drops “not” into the next chunk, the vector mostly sees “refund.” Dimensionality is often a few hundred to a bit over a thousand; similarity is not a calibrated probability. Suggested order: filter on policy id and effective date, use a sparse scorer (for example BM25) to catch surface cues like “not / do not / forbidden,” then rerank with a cross-encoder for actual answerhood. If you inject negation pairs into the dual encoder, you can push every sentence containing “not” away and collapse recall. Near in embedding space ≠ true; treat it as a candidate generator only.

**Follow-ups:** Do instruction embeddings (different prefixes on query vs document) mitigate negation, or do you still need a reranker; if chunks grow from ~200 to ~800 characters, how does the chance of cutting off the negation change; when should you drop vectors and do symbolic lookup on the policy store.

**Extended questions:**
- Why can you almost never use the LLM’s token embeddings—or the mean of its last layer—as a retrieval index?
- In multilingual refund policies, negation morphemes differ—do bilingual neighbors get worse than monolingual ones?
- How would you regress with a set of “negation-only” adversarial queries so an embedding upgrade cannot look better on recall while getting logically worse?

**Reasoning:** Read off what pairing objective the embedding actually optimized; negation pairs are then out-of-objective, not a cosine bug, so fix filtering and reranking first. Accept with precision@k and false-neighbor rate on those adversarial queries.

## ARCH-013. Context grows from 2k to 8k. For one self-attention layer, by about what factor do prefill compute and decode KV reads grow, and do you trim the prompt or change attention first?

**Answer:** Length ×4. The quadratic attention terms (QKᵀ and the weighted sum over V) follow n², so prefill is about ×16; QKV projections and the FFN follow n, about ×4. Each decode step still has one new query, so compute is roughly linear in n, but the KV volume you read is also ×4 and bandwidth often breaks first. The full n² hit appears in prefill (and in a loop that wrongly recomputes the whole sequence); correct decode folds history into the cache. Trim or summarize the prompt first, because n feeds both the quadratic term and KV; only if the prompt is already the minimum necessary set do you add GQA/quantization or windowed attention—do not start by swapping the backbone.

**Deep dive:** Let head dim be Dh, heads H, `D=H·Dh`. Prefill attention FLOPs are on the order of `~ 2n²D` (plus projections `~ 4nD²`; constants vary by whether you count the output proj). For n: 2k→8k the quadratic term is ×16 and projections ×4; at long context the quadratic term can go from secondary to dominant. KV bytes are about `2 × L × n × H_kv × Dh × sizeof`; if `H_kv=H` that is about `4D` bytes per token per layer before the dtype multiplier. Illustrative: width 4096, 32 layers, fp16, full-head cache is on the order of 0.5MB per token, so 8k is a few GB—check actual `H_kv` and precision. GQA that drops `H_kv` from 32 heads to 8 cuts KV about ÷4 and decode read bandwidth with it. A sliding window caps effective k_len and gives up one-shot long-range visibility: fine for local logs, bad for “policy at the start, question at the end.”

**Follow-ups:** If average history is 1k vs 8k, where is the inflection between “trim prompt first” and “GQA first”; when windowed attention would drop the system prompt, do you pin it as an always-visible anchor or re-insert it; can block-sparse or linear attention replace prefill’s quadratic term without also dropping decode bandwidth.

**Extended questions:**
- As batch size goes from 1 to 32, does decode become compute-bound or KV-bandwidth-bound first?
- “Lost in the middle” at 8k—is that an n² compute problem or an attention-allocation problem?
- For a given n and D, where do the FFN’s O(n) and attention’s O(n²) cross inside one layer?

**Reasoning:** Write the goal as “keep the quality of n×4 without paying ×16 latency/memory,” split the quadratic compute curve from the linear bandwidth curve, and reduce n (prompt) before reducing bytes per token (GQA/quant). Verify with prefill ms, decode tok/s, and KV occupancy—not the advertised window size.

## ARCH-014. The same service must do bidirectional retrieval encoding and dialogue generation. Why can’t you deploy decoder-only only?

**Answer:** A decoder-only stack uses a causal mask, so position i never sees tokens after i. Sentence vectors then rely on the last token or one-way pooling, and symmetric similarity is usually weaker than a bidirectional encoder of the same size. Generation is the opposite: an encoder-only model has no autoregressive head and cannot decode token by token; an encoder-decoder can do conditional generation but keeps two stacks, which is heavy for chat where context and reply share one causal chain. Retrieval wants bidirectional attention; generation wants causal attention. Those patterns fight, so one causal weight set pays a clear quality tax on one of the two jobs.

**Deep dive:** Encoder-only: fully bidirectional self-attention, typically a masked or contrastive objective, then CLS / mean pooling over [B, S, d]. There is no decode loop and no KV cache that grows with generated steps. Decoder-only: causal mask plus next-token prediction; after prefill, decode reuses KV—this fits chat and tool loops. Encoder-decoder: the encoder reads the source bidirectionally; the decoder runs causal self-attention on the target so far and cross-attends to encoder memory. When source and target are not aligned (translation, summarization), cross-attention is stabler than concatenating both into one causal string. Magnitudes: retrieval QPS is often much higher than generation QPS, so a ~100–300M bidirectional encoder for vectors plus a 7B-class decoder for text usually beats asking one 7B causal model to moonlight. At the same depth, encoder-decoder parameters are about 2× a single stack, or each stack is half as deep. Edge case: embedding fine-tunes on a causal model (contrastive loss plus a “compress this sentence” instruction) can approach bidirectional quality, but you still attach another head or LoRA—two behaviors, not a free combo.

**Follow-ups:** Can one Prefix-LM checkpoint (bidirectional prefix, causal suffix) cover both jobs? Do cross-attention K/V need a cache, and how does that differ from self-attention KV cache? What do failure cases look like when you retrieve with a causal last-token embedding?

**Extended questions:**
- For 512-token passages and 20-token queries, how much do attention FLOPs differ between a bidirectional encoder and a causal decoder?
- If you must ship a single model, what do you pay in latency and KV to use an encoder-decoder for multi-turn chat?
- If retrieval width and the generator hidden size differ, where in the service do you project?

**Reasoning:** Start from the information direction each task needs: retrieval is symmetric and bidirectional; generation is causal and cacheable. Masks and output heads decide which stack can do which job. Then compare “one large causal model moonlighting” with “small bidirectional encoder + generator” on QPS and quality. Accept with retrieval recall and dialogue scores as a pair.

## ARCH-015. When you sketch a decoder-only inference graph, which of causal mask, RoPE, KV cache, and GQA break correctness if removed, and which only change speed?

**Answer:** Removing these breaks autoregressive correctness: token embeddings, a causal mask (or an equivalent “see only the past” constraint), the stacked attention + FFN + residual + norm, and the LM head. If the weights were trained with RoPE, dropping it at inference puts Q/K off the trained distribution and outputs break—a correctness error, not a quality tweak. Only a model trained from scratch without explicit position encoding (NoPE, relying on the causal mask) can still run, with weaker sentence order, copying, and length extrapolation. KV cache and FlashAttention are inference implementations: they should match recomputation numerically; removing them should only slow you down. GQA/MQA is a head-sharing choice fixed at training time—you cannot flip it at serve time or load one as the other (shape mismatch or a silent mis-slice), but an MHA checkpoint can be converted to GQA by mean-pooling K/V heads within each group into one head, then a brief uptraining (a small fraction of the original pretraining compute, as in the GQA paper, Ainslie et al. 2023). Calling all four “architecturally mandatory” misleads debugging: a wrong mask is a right/wrong bug; a wrong cache is first a speed and numerical-equivalence bug.

**Deep dive:** Shape contract: hidden state `[B, S, D]`, Q after the head split `[B, H, S, Dh]`, and under GQA K/V may be `[B, H_kv, S, Dh]` with `H_kv < H`. Prefill needs a lower-triangular mask so position i cannot read past i; if decode dots one current query against cached K, the mask is implied by what lives in the cache. Residuals make depth trainable. Modern stacks often use Pre-Norm + RMSNorm and a gated SwiGLU FFN—expressivity and stability choices, not “next-token is undefined without them.” Weight tying of the LM head and embedding is optional. MoE replaces the FFN and is also not skeleton. Failure map: missing causal mask → train loss looks too good, decode disagrees with train; missing KV cache → answers should be nearly unchanged but every step recomputes the prompt; RoPE-trained weights served without RoPE → wrong outputs; trained NoPE (no RoPE/ALiBi/relative bias) → swapped sentences and “k-th item” tasks degrade; loading GQA weights as MHA → hard error or a silent mis-slice.

**Follow-ups:** Does Pre-Norm → Post-Norm hurt training stability first or inference numerics; how far a no-explicit-PE (sometimes called NoPE) causal model can go; how you prove cache equivalence by comparing per-token logits with the cache toggled on the same input.

**Extended questions:**
- Is multi-head an expressivity knob or a correctness requirement? What can a single-head decoder still do, and where does it fall over?
- Why does mean-pooling MHA KV heads into GQA without uptraining lose quality, and what does the short uptraining recover?
- How does putting norm before vs after attention change gradient scale and how deep you can stack?

**Reasoning:** Label each piece as “without it, is the next-token conditional still well-defined / still trainable / only a throughput change,” then back the label with one concrete failure. Accept with a mask unit test, logits alignment with cache on/off, and an ablation on position-sensitive tasks.

## ARCH-016. If you keep hidden size fixed and replace 32 heads with one wide head, where do expressiveness and KV bandwidth each take the hit?

**Answer:** One head shares a single attention map across the sequence and can learn only one “who attends to whom” pattern. Multi-head splits the same hidden size into subspaces, e.g. 32×128, so heads can simultaneously track syntax, coreference, proximity, then concatenate back. QKV parameter count barely moves under that reshape. KV-cache bytes scale with KV heads × head_dim: if every query head still owns a KV pair, 32→1 cuts bandwidth about 32× and usually hurts expressiveness a lot. That is why GQA/MQA keep many Q heads and fewer KV heads.

**Deep dive:** Head h computes softmax(QₕKₕᵀ/√dₖ)Vₕ, then concat and Wᴼ. dₖ = d_model / n_heads, e.g. 4096/32=128. Heads do not share the attention matrix, which is the point. More heads are not free: head_dim below about 32 weakens the dot product; too many heads go redundant. GQA shares K/V across a group of query heads, so cache grows with KV heads, not query heads: 32 query / 8 KV is about one quarter of full MHA. Swapping a 32-head MHA checkpoint to 8-head GQA without training it that way is an architecture mismatch; if coreference drops, check the training layout before blaming “not enough heads.”

**Follow-ups:** Why head_dim is usually 64 or 128 rather than 16; how much bandwidth MQA saves on long context and which alignment tasks it tends to hurt; how you would visualize whether heads actually specialized.

**Extended questions:**
- Write per-token KV bytes for MHA, GQA, and MQA, and plug in 32 query heads, 8 KV heads, dₖ=128, FP16.
- If one head dies or explodes, how can Wᴼ absorb that into the residual stream?
- If inference bandwidth is the hard cap, which tasks must you score before choosing a head layout?

**Reasoning:** Hidden size is fixed; the knobs are “how many independent alignments” versus “how many KV tensors.” Expressiveness tracks the former, bandwidth the latter times precision. Split those two, then choose MHA versus GQA, and accept only if coreference, long-range lookup, and throughput all hold.

## ARCH-017. At generated position t, how many rows do Q, K, and V have? If the score is wrongly Q·Qᵀ, which projections become dead weights?

**Answer:** On a correct decode step, Q has one row (the current token); K and V have t rows (past plus current). They are linear maps of that layer’s normalized hidden state through `W_Q, W_K, W_V`, then split into heads. Scores should be `QKᵀ / √Dh` with shape `[B, H, 1, t]`, then a weighted sum of V. If the training graph uses `QQᵀ` for scores and V still comes from `W_V`, `W_K` never enters the graph and its gradient should be ~0—a dead weight. `W_Q` then plays both query and “fake key,” and the retrieval split is gone. Note: swapping the names of K and V when training from scratch is often just a reparameterization and loss can still fall—define the bug as “who never entered the dot product,” not “the variable names are crossed.”

**Deep dive:** Shape contract:

```
Q: [B, H, q_len, Dh]   # decode: q_len=1; prefill/train: q_len=S
K: [B, H_kv, k_len, Dh]
V: [B, H_kv, k_len, Dh]
scores: [B, H, q_len, k_len]
```

Q is “what this position is looking for,” K is “the index each past position offers,” V is “the content being aggregated.” `1/√Dh` rescales the dot product so softmax does not saturate. A mistaken `QQᵀ` at train time still has scores `[B, H, S, S]`, so a shape unit test can pass—you have to watch `W_K`’s grad norm or whether the parameter updates. Under GQA, `H_kv < H`; K/V still have `k_len` rows, just fewer heads, and query heads share groups. Boundary: prefill’s Q also has S rows, but that is one parallel shot, not decode rebuilding Q with t rows every step. Computing full-sequence Q at decode and slicing the wrong score row shows up as mis-aimed attention plus wasted FLOPs.

**Follow-ups:** Why swapping V and K during training can still converge; what you lose if there are no separate `W_Q/W_K/W_V` and attention uses the same x (an older variant); how one attention map tells you whether the model is copying context or retrieving with the current query.

**Extended questions:**
- If Dh grows from 64 to 256 and you forget to scale the dots, what numerical symptom appears in softmax first?
- In cross-attention, whose length sets Q’s rows and whose length sets K/V’s rows?
- If you apply RoPE only to K and forget Q, does relative position still hold?

**Reasoning:** From “this step adds one position,” write the row contract, then substitute the bad formula into the graph and see whose gradient vanishes. Accept with shape asserts plus a `W_K` gradient probe, not with “loss still went down.”

## ARCH-018. After widening head_dim from 64 to 256 the attention collapses to one-hot. How do you use the variance of the dot product to decide whether to scale by √dₖ?

**Answer:** Start by estimating the scale of Q·K. If coordinates are roughly zero-mean, variance about 1, and weakly dependent, the unscaled dot product has variance about dₖ: going from 64 to 256 grows the standard deviation from about 8 to about 16. Huge positive and negative scores push softmax into saturation, weights go nearly one-hot, and gradients on the logits vanish. Dividing by √dₖ brings that variance back to order 1 so attention stays on a trainable slope.

**Deep dive:** With q,k ∈ ℝ^{dₖ} and q·k = Σᵢ qᵢkᵢ, independence plus unit variance gives Var(q·k)=dₖ, so √dₖ is a constant-scale reset of the scores that enter softmax. Real Q and K after a linear map need not have variance 1; some stacks also multiply by a learned scale or RMSNorm Q/K, so “must divide by √dₖ” is not the only stabilizer. Still, growing dₖ with no scaling is the usual way to saturate. The softmax Jacobian carries pᵢ(1−pᵢ), which dies as p→0 or 1. Entropy of a layer’s attention near 0 flags collapse; the gradient norm on QK is the cross-check. Typical head_dim is 64 or 128; 64→256 with the old scale is roughly halving the temperature and makes training brittle.

**Follow-ups:** If Q and K are already RMSNormed per head, do you still divide by √dₖ; when a learned scale and √dₖ coexist, which one dominates; how do attention entropy and gradient histograms separate “under-scaled” from “learning rate too high.”

**Extended questions:**
- Under the i.i.d. unit-variance assumption, write Var(q·k) and the variance after dividing by √dₖ.
- If the true activation variance is σ² rather than 1, what should the scale factor become?
- How would you dump percentiles of one layer’s attention logits from a single forward pass to confirm saturation?

**Reasoning:** The goal is a stable softmax input. The constraint is that a wider head sums dₖ products, so variance grows linearly. Write the variance, then the saturated gradient, then match entropy and grad logs. If Q/K are already normalized, re-estimate variance before keeping or dropping the scale; do not memorize the coefficient.

## ARCH-019. A back-of-the-envelope count shows most 70B dense parameters sit in the MLPs. What does that imply for widening the FFN, adding layers, or replacing the FFN with MoE?

**Answer:** Attention QKV plus the output projection is about 4d² per layer. A two-matrix MLP with expansion ~4 is about 8d²; SwiGLU uses three projections with an inner width often around (8/3)d to 4d, and the FFN still usually holds most of the layer. Widening the FFN is the most direct capacity knob. Adding layers copies both attention and FFN and also lengthens the KV cache. MoE copies that heavy FFN into many experts and runs a few per token, trading almost-constant compute for a much larger parameter pool.

**Deep dive:** After attention has mixed context into each position, the FFN is applied independently: a common SwiGLU is SiLU(xW_gate) ⊙ (xW_up), then W_down. It does not look at neighbors; it recombines channels and stores mapping-like capacity. Illustrative scale: d=8192 and inner 28672 gives about 3×8192×28672 ≈ 7.0×10⁸ FFN weights per layer; 80 layers is about 56B, already the bulk of a 70B-class dense stack. Quantization, sparsity, and expert parallelism therefore hit the FFN first. Limits: a very wide FFN blows training activation memory; MoE adds routing imbalance, idle experts, and cross-device traffic. The FFN is not a small tail after attention—it is the main capacity bin.

**Follow-ups:** Why expansion often sits near 4 rather than 16; how to convert parameters and activations when SwiGLU adds a third matrix versus a two-layer GELU MLP; under a fixed inference FLOP budget, whether a wider dense FFN or more experts is the better buy.

**Extended questions:**
- For d=4096, 32 layers, SwiGLU inner width 11008, roughly what fraction of parameters is FFN?
- If a single FFN step cannot see neighbors, how does distant information arrive after many layers?
- After switching the FFN to MoE, which routing stats do you watch to catch expert collapse?

**Reasoning:** Split a layer: ~4d² attention versus ~8d² or three-projection FFN, so capacity lives in the FFN. If you want more knowledge under a tight decode budget, MoE-ize the FFN. If you want richer mixing, deepen the attention stack. Bound width with both the parameter formula and activation memory, not the headline parameter count alone.

## ARCH-020. Replace LayerNorm with “divide by RMS, skip mean subtraction, skip bias.” Which forward steps disappear, and why can the deep residual stream still stay on scale?

**Answer:** RMSNorm computes x̂ = x / √(mean(x²)+ε) and multiplies by a learned γ; common implementations omit β. Versus LayerNorm you drop one mean, one vector subtract, and one add-bias; the reduction is still over D, so the kernel is shorter and touches less memory. It does not pin per-dimension mean to 0 — only the Euclidean scale of the vector — so the residual stream does not inflate linearly with depth. A shared additive offset has to be absorbed by the next linear layer. Many modern decoders pair it with Pre-Norm because those skipped ops cost almost no stability.

**Deep dive:** For x∈R^D, LayerNorm stores μ, broadcasts a subtract, then a variance; RMSNorm does one mean-of-squares. At D=4096 that is several full-width reads/writes fewer per token per layer, which is visible in bandwidth over tens of layers and long sequences. The scale target differs: LayerNorm pins a near-standard-normal shape; RMSNorm pins only the second moment, so the first moment may be nonzero. If every channel shares a large additive bias, RMSNorm will not clear it, but the next W can learn to cancel. Keep the reduction in FP32 and init γ at 1 so the first layer does not immediately amplify activations. This is not “LayerNorm for free”: if a unit test assumes per-dimension mean ≈ 0, RMSNorm output will fail it — do not copy the assertion.

**Follow-ups:** Why do so many implementations delete β but keep γ? Once the residual path is already Pre-Norm, does mean subtraction still earn its keep? Stacked with a gated FFN (e.g. SwiGLU), which op owns scale and which owns sparsity?

**Extended questions:**
- Write a three-line RMSNorm and line it up against LayerNorm to mark the dropped operators.
- Under mixed precision, lifting only the RMS reduction to FP32 and leaving linears in BF16 usually kills which class of NaNs?
- If γ is frozen at 1, is deep training more likely to blow up in gradients or in activation amplitude?

**Reasoning:** The goal is a stable deep scale; the constraint is fewer ops and less memory per layer. Split LayerNorm into center / divide / affine, drop the mean-center and β that the residual path does not need, and keep divide-by-RMS plus γ. “Control the second moment; let the next linear eat bias” explains why training still works. Check that activation RMS does not explode with depth and that the loss curve stays smooth after the swap — not that the mean is zero.

## ARCH-021. When decoding the t-th target token in translation, which sequence do Q, K, and V each read? Why can the source side be cached in full, and what breaks if you also put a causal mask on the source?

**Answer:** At step t, Q comes from the decoder’s current-layer hidden state of length 1 (the current target position); K and V come from the encoder’s output over the whole source sentence, with S_src rows, independent of t. The formula is still `softmax(QKᵀ/√Dh)V`, but scores are `[1, S_src]` (or `[S_tgt, S_src]` at train time). The source side usually has no causal mask: translating token t is allowed to see the entire source. The source does not change after one encode, so source K/V are computed once and reused every decode step—unlike self-attention KV, which grows with t. If you wrongly apply a causal mask to the source, later source tokens become invisible to earlier target positions: the second half of the sentence is blocked, alignments bias toward the front, and the tail is dropped. A text-only decoder-only chat model has no second heterogeneous sequence; prompts and retrieved text are the same kind of token and belong in causal self-attention, so a dedicated cross-attention layer is unnecessary.

**Deep dive:** Self-attention takes Q, K, V from one sequence; cross-attention takes Q from the target and K/V from the source. An encoder-decoder layer typically runs causal self-attention on the target (already generated translations only), then crosses into encoder memory. Source-cache lifetime: frozen when encode finishes, not appended as new target tokens arrive; target-side self-attention KV appends one row per step. Incremental cost per step is about O(S_src · D), so a very long source bottlenecks on S_src and wants chunked encoding or compressed memory—not blame on target length. The same pattern appears whenever a fixed condition modulates another sequence, for example text features as K/V and another modality as Q; if the condition is fixed it is encoded once. Boundary: concatenating source and target into one self-attention stream can also do conditional generation, but the quadratic term follows (S_src+S_tgt)² and the source prefix occupies every layer’s self-attention cache. That is a different implementation; you cannot “turn on” cross-attention at inference on decoder-only weights—the graph must match training.

**Follow-ups:** How train loss and omission patterns differ for “future leak on the target side” vs “source wrongly causal-masked”; with source 4k and target 128, which is larger, cross-attention FLOPs or self-attention; if the user edits the source mid-dialogue, which caches must be dropped.

**Extended questions:**
- Which cache grows with generate steps, and which freezes after encode—source K/V or target self-attention KV?
- When the source is long, is the bottleneck n_q or n_kv, and how do you compress source memory?
- If the decoder must revise the source over several passes (not just look it up one way), is a single unidirectional cross-attention layer still enough?

**Reasoning:** Fix “what is being generated vs what is a read-only condition,” so Q follows the target step and K/V follow source length and can freeze; then reject a source causal mask because every source token should stay visible. Pure text has no second source, so do not add a layer for a prettier diagram. Accept on score shape `[S_tgt, S_src]`, source cache not growing with t, and whether omissions cluster in the source tail.

## ARCH-022. Training loss is already low, but autoregressive generation is incoherent. How do you use a mask A/B test to tell whether future tokens leaked?

**Answer:** The cleanest check is the same batch under a strict lower-triangular causal mask versus a fully visible mask, comparing next-token loss at every position. If the bidirectional run is much cheaper and generation still falls apart, training was copying future tokens and the loss no longer stands in for left-to-right decoding. A causal decoder must let position i see only ≤i, with labels shifted by one from the inputs.

**Deep dive:** Implementations usually add −∞ to the upper triangle of QKᵀ so those softmax weights are 0. Training is parallel: [t₀…t_{n−1}] predicts [t₁…tₙ]. If hidden state i can read t_{i+1}, the model takes a shortcut, eval loss can look great, and generation—where the future is absent—breaks. Typical leaks: flipped mask, wrong shift, unmasked padding, or cross-document visibility after packing. An attention heatmap should show a hard-zero upper triangle; the gradient of position-i loss w.r.t. a future embedding should be 0 if nothing leaked.

**Follow-ups:** Why a bidirectional encoder wants both sides; in which order padding and causal masks are written into the score matrix; when packing many documents, what a block-diagonal mask blocks that a purely causal mask does not.

**Extended questions:**
- Draw a length-5 causal mask and mark which keys position 2 may see.
- If you add the causal mask only at inference and forget it in training, what loss-versus-generation pattern shows up online?
- How would you turn “gradient to future embeddings is zero” into an automated regression?

**Reasoning:** Autoregression’s hard constraint is “no future at generation time.” Training loss transfers only when the visible range matches that constraint. Flip the mask for an A/B, then inspect shift and padding. Low loss plus bad generation is a leak until proven otherwise—not a temperature bug.

## ARCH-023. If you replace residuals in a 32-layer Transformer with overwrite updates, where does training break first, and how do gradients travel?

**Answer:** A residual writes each sublayer as y = x + F(x), so the backward pass is ∂L/∂x = ∂L/∂y · (I + ∂F/∂x); the identity term is a gradient path that skips attention and the FFN. If you switch to y = F(x), 32 Jacobians multiply, gradient norms decay or explode exponentially with depth, and loss spikes or NaNs usually show up first in the deeper blocks. Residuals are not a decorative extra edge: they keep deep layers receiving a gain near 1 and stop lower-layer features from being fully rewritten at every block.

**Deep dive:** In the usual Pre-Norm block the data flow is x ← x + Attn(Norm(x)), then x ← x + FFN(Norm(x)); the skip is identity and the shape stays [B, S, d_model]. Even if a sublayer Jacobian has singular values σ < 1, the identity still forwards the gradient unchanged. At σ = 0.9, σ³² ≈ 0.03; at σ = 0.7, σ³² ≈ 1×10⁻⁵, so a pure overwrite path is already almost disconnected at 32 layers. Post-Norm puts normalization after the add and scales that identity path away, which is why stacks deeper than about 20 layers are hard to train that way and modern LLMs put LayerNorm or RMSNorm before the sublayer. Edge cases: if F(x) and x differ by several orders of magnitude the residual itself drifts, so people zero-init the residual branch or use layer scale. At inference a residual is one elementwise add, not a latency bottleneck. Do not confuse it with concatenating two streams and projecting; that cuts the free unit-gain path.

**Follow-ups:** Why does Pre-Norm preserve the residual identity better than Post-Norm? What problem does a learnable residual scale solve? Can a longer LR warmup stabilize 32 layers if you drop residuals entirely?

**Extended questions:**
- If the residual becomes y = W[x; F(x)], how do parameter count and the backward path differ from an identity skip?
- In an 80-layer model, do embeddings mainly ride the residual upward, or does each layer rewrite them?
- If F(x) stays near 0 late in training, what did that sublayer learn, and can you prune it?

**Reasoning:** The goal is trainable depth and reusable representations; the constraint is that backprop multiplies Jacobians. Compare y = F(x) with y = x + F(x), estimate gain after 32 multiplies via σ³², and you get both “breaks in the deep layers” and why Pre-Norm keeps a clean identity. Confirm with per-layer gradient-norm curves and the layer index of the first NaN.

## ARCH-024. BatchNorm on a variable-length decoder batch makes loss jitter as soon as padding changes; normalizing each position’s hidden vector steadies it. Along which axis are mean and variance computed, and why does decode not need a running mean?

**Answer:** LayerNorm computes mean and variance on the d_model vector of one sample at one position, then (x−μ)/(σ+ε), then γ and β. On a [B, T, D] tensor the reduction is the last axis D — not other sentences in the batch, not other tokens in the same sentence. BatchNorm reduces over the batch (or batch×time), so pad fraction, sentence length, and decode at B=1 all desynchronize statistics from training. LayerNorm’s statistics come from the vector itself, so training and step-by-step decode stay in the same distribution. That is why variable-length language models use it instead of the running-mean BatchNorm common in vision.

**Deep dive:** γ and β have shape [D] so the layer can restore per-channel scale and shift after the normalize; without them every sublayer is pinned at mean 0, variance 1. The mean/variance reduction is often kept in FP32 under mixed precision: D is typically thousands, and summing many small half-precision values then subtracting a mean underflows or NaNs. With no norm at all, the residual stream accumulates and activation/gradient scale drifts with depth. Versus RMSNorm, LayerNorm subtracts the mean then divides by the standard deviation; RMSNorm only divides by RMS. Versus BatchNorm, the axis is the difference. Mask pads before the loss, but LayerNorm should not mix pad values into other tokens — and it does not, which is exactly why it fits ragged sequences.

**Follow-ups:** If someone normalizes along T (one μ, σ shared by all positions in a sentence), what happens to those statistics during autoregressive decode? After γ=1, β=0 init, what is early training actually constraining? Why must the LN reduction run at higher precision while neighboring linear layers can stay BF16?

**Extended questions:**
- Write LayerNorm on [B, T, D] and mark whether pad positions should enter μ and σ.
- Train a tens-of-layers decoder with every norm removed: where do loss and grad-norm usually blow first?
- Can LayerNorm and RMSNorm weights on the same checkpoint be swapped? Which parameters are the mismatch?

**Reasoning:** The symptom is “loss jitters when batch composition changes,” so ask which examples the statistics depend on. BatchNorm depends on other sequences in the batch, so variable length and B=1 must drift. Reducing over D makes each position self-contained; decode needs no running mean. Then use FP32 reductions and γ/β for numerics and capacity. Acceptance is train/decode statistic agreement, not only that training loss still falls.

## ARCH-025. A near-hundred-layer Post-Norm run explodes in grad-norm halfway through; switching to “normalize, then sublayer, add the residual as-is” shortens warmup. Which Jacobian did the gradient stop multiplying through?

**Answer:** Post-Norm is x←Norm(x+Sublayer(x)): both the residual and the sublayer output go through another normalize, so shallow gradients are a product of many Norm Jacobians and tens-to-hundreds of layers blow up unless learning rate crawls for a long warmup. Pre-Norm is x←x+Sublayer(Norm(x)): the residual branch is identity, so gradients can reach the embedding without passing through Norm. What you stop multiplying is the derivative of the Norm that used to sit after each residual add. Deep decoders typically use Pre-Norm plus one final norm after the last block, because the residual stream itself is no longer rescaled.

**Deep dive:** In backward, Post-Norm’s main path is ∂Norm/∂z · (I + ∂Sublayer/∂x); Norm under a mean–variance constraint shrinks gradients, and with depth the embedding updates for early tokens become unstable. Pre-Norm adds a pure +1 path and superposes the sublayer gradient, so warmup can be much shorter. The cost is that the residual only accumulates: later layers’ increments look small relative to the stream, and at the same depth a well-tuned Post-Norm sometimes wins, which is why DeepNorm (up-weight residual, down-scale sublayer init) and sandwich norms exist. Diagnose two things: whether grad-norm spikes in the first few thousand steps, and whether last-layer output RMS inflates monotonically with depth. Skip final norm on Pre-Norm and logit scale drifts, so softmax becomes peaky or flat. “It trains” is not “best at this depth.”

**Follow-ups:** Why does Post-Norm specifically need a long warmup rather than only a smaller global learning rate? If you drop the final norm on Pre-Norm, which metric breaks first — loss, grad-norm, or sampled repetition? Do DeepNorm and sandwich buy stability or capacity?

**Extended questions:**
- Draw both blocks and mark which backward path is identity and which goes through Norm.
- If you must train ~100 layers Post-Norm, besides a longer warmup, which init coefficients move?
- How do you use per-layer grad-norm curves to blame Norm placement rather than data noise?

**Reasoning:** The failure is “depth plus exploding grads”; the constraint is that the residual must still connect the bottom to the output. Write both formulas and look only at whether backward crosses Norm: Post-Norm multiplies an extra Jacobian per layer, Pre-Norm’s residual is I. That predicts a shorter warmup and an unbounded residual scale, so you add final norm. Validate with grad-norm and output RMS, not only final perplexity.

## ARCH-026. After you drop learned absolute position embeddings and keep only the causal mask, can the model still reliably notice that two instruction sentences were swapped?

**Answer:** It can notice in part, but that is not a stable position system. Bidirectional self-attention is permutation-equivariant; a causal mask lets the later sentence see the earlier one and not vice versa, so a swap changes who can read whom and theoretically still carries order. The signal is weaker than explicit positions: intra-sentence word order, absolute k-th items, and two parallel instructions of similar length smear easily. In production, relative schemes such as RoPE beat stretching an absolute embedding table because rotations do not grow a `max_len` table and usually extrapolate past the training length more controllably (still measure it—this is not infinite extrapolation).

**Deep dive:** Position can enter along at least: an absolute or sinusoidal table added to embeddings; RoPE rotating Q/K so scores depend on relative offset; biases such as ALiBi subtracted from scores; and the causal triangle itself. With only the mask, swapping two sentences (content changes index) flips visibility, so “it cannot see order at all” is false. Shuffle three numbers inside one sentence, or ask for “the second ID from the end,” and the mask does not give a global coordinate. Absolute embeddings hard-stop at `pos < max_len`; beyond that you interpolate or error. RoPE writes position into angles, so a model trained to 4k can sometimes reach a bit further, but extreme jumps (for example ×8) still collapse and any extra scaling trick needs its own eval. Ablate with a task “swap two instructions, the output must follow,” not language-modeling loss alone.

**Follow-ups:** If the causal mask already breaks permutation equivariance, why most LLMs still want RoPE; on “copy the 17th number in context,” who is stabler, relative or absolute; when length extrapolation fails, do you look at attention-distance histograms first or at perplexity.

**Extended questions:**
- Train 4k, infer 32k: which failure shows up first under absolute-table interpolation vs RoPE extrapolation?
- If you raise or lower RoPE base frequencies, who flinches first—short-sentence syntax or long-document retrieval?
- How does dropping position encodings differ for a bidirectional encoder vs a decoder-only stack?

**Reasoning:** Ask what asymmetry remains without explicit positions, credit the causal mask’s directionality, then show it is not enough on tasks that need absolute indices—that is the engineering case for RoPE (relative offset, no table growth). Accept with swap / k-th item / long extrapolation cases.

## ARCH-027. A model trained at 4k is fed 16k and the absolute position table goes out of range. After rotating Q and K by position, what position relation is left in the dot product, and how far can you extrapolate before it becomes unstable?

**Answer:** A learned absolute table has L_max rows; an index past 4k has no row. RoPE never looks that table up: it rotates each head’s Q and K in 2-D planes by angle m·θ_i (m is the position), so Q_m·K_n depends on the relative offset m−n and on content — there is no “row 16000.” Relative phase is reliable inside the trained span; a raw 2–4× stretch sometimes still computes. Farther out, high-frequency planes spin too fast and distant scores flatten, so you usually interpolate or retune the base frequency and run some long-context training. The rotation formula alone is not an extrapolation license.

**Deep dive:** On paired dimensions the rotations satisfy R_m⊤R_n = R_{n−m}, hence q⊤R_m⊤R_n k = q⊤R_{n−m}k and the absolute index cancels. Frequencies are often θ_i = 10000^{−2i/d} (a common base, illustrative): low-frequency planes turn slowly and carry long range; high-frequency planes turn quickly and carry adjacency. Absolute tables fail by out-of-range indexing or by reusing the last row on longer positions — both leave the training distribution. RoPE fails differently: beyond the trained offset, distances that should be distinct wrap in the fast planes and attention looks almost uniform. Position interpolation, a larger effective base, or piecewise scaling pull 16k phases back into a 4k-seen range, then you continue training on long documents. Judge the radius by whether long-context retrieval or continuation still aligns, not by whether the formula still returns a number.

**Follow-ups:** If you instead add an absolute vector onto the embedding, which step breaks first under length extrapolation? ALiBi encodes distance with a linear bias — compared with RoPE, which is calmer at extreme length, and which is more sensitive to exact offsets? When applying YaRN/NTK-style scaling, which end of the spectrum do you move for low versus high frequencies?

**Extended questions:**
- Write the 2-D case of Q_m·K_n so that only m−n remains, and say why d_head must come in pairs.
- Without changing weights, only RoPE frequencies, you attach a 4k model to 16k. Which probes (copy nearby, copy far, needle retrieval) do you run first to see if it collapsed?
- After inserting a span, must later tokens’ phases be recomputed as a block, and what happens to the KV cache?

**Reasoning:** The goal is to still score attention at 16k; the constraint is no unseen absolute index. Compare table lookup with rotating phase, use R_m⊤R_n=R_{n−m} to show the dot product keeps only the relative offset, then use fast versus slow frequencies to estimate the risk of a raw 4× stretch. Separate “the formula runs” from “phase is still in-distribution,” and only then decide on scaling plus continued training.

## ARCH-028. An edge device must stream a 32k-context transcript and almost cannot afford memory that grows linearly with length. Why would you seriously consider a linear recurrent layer instead of standard self-attention? What do you give up at training time?

**Answer:** Standard self-attention lets each new position meet every past position in one hop, but K/V cache memory and stepwise attention read bandwidth both grow with n; a 32k stream often hits memory first. A linear recurrent layer (including state-space forms) updates a fixed-size hidden state each step, so inference memory is roughly O(1), which matches “emit while listening, do not store a full KV.” The cost is stronger sequential dependence during training and long-range signals that must squeeze through a fixed-width state instead of jumping in one hop.

**Deep dive:** RNN-style units unroll as h_t = f(h_{t−1}, x_t); information walks one step at a time, early gradients vanish, and gates in LSTM/GRU help but still block full time-parallel training. Transformers train the whole span in parallel—that, not a theoretical monopoly on sequence modeling, is why they absorbed large corpora. Linear recurrence / state space makes f a linear scan over the input, so a prefix scan can restore training parallelism while inference keeps a constant state. Hybrid blocks (some attention, some linear) are common: local or sparse attention restores precise short-range alignment, linear layers carry the long-context memory bill. For offline, batched training with spare memory, full attention is still simpler and the tooling is more mature.

**Follow-ups:** Why does a recurrent state not need a separate position encoding, while self-attention usually does? At 32k, what kind of information does a fixed-width state drop first? How do you keep a training-time prefix scan aligned with step-by-step state updates at inference? If you must keep a few attention layers on the device, do you put them shallow or deep?

**Extended questions:**
- For a 32k input, sketch how memory grows for a full KV cache versus a constant-size state, and name the constants still hiding in the O(1) (layers, heads, state width).
- Autoregressive generation is serial even inside a Transformer. Is that the same serial story as RNN training?
- Which product constraints (battery, max session length, permission to drop early audio) would send you back to sliding-window attention instead of linear recurrence?

**Reasoning:** The goal is streaming 32k; the constraint is memory that almost does not grow with n. KV growth under standard attention hits that constraint directly. A fixed-state recurrent layer meets the memory budget and turns long-range interaction into “can it fit in the state.” Training parallelism is the second constraint: only a linear scan that can be re-parallelized at train time is a serious choice in a modern stack.

## ARCH-029. Your decode loop reruns the entire prompt every step — which tensors should be cached, and by roughly what factor does fixing it cut attention compute?

**Answer:** Prefill consumes the full prompt once at shape `[B, S, D]`, writes each layer’s K and V for all positions into cache at about `[B, H_kv, S, Dh]`. Each later decode step feeds only the new token, builds Q with `q_len=1`, attends to cached K/V, and appends this step’s k,v; the FFN also runs at that single position. If every step instead runs full-sequence self-attention at length `S+t`, score compute degrades from “one query × current length” to “current length².” For an illustrative ~2k prompt plus 128 new tokens, the score term alone drops by about two to three orders of magnitude after the fix (roughly 2000×, linear in current length). The FFN also falls from O(S) per step to O(1).

**Deep dive:** Inside a layer, Q/K/V are typically split to `[B, H, q_len, Dh]`. Correct decode has `q_len=1`, `k_len=S+t`, scores `[B, H, 1, S+t]`. The buggy loop is `q_len=k_len=S+t` every step. Magnitudes: S=2048, T=128; the bug costs ~(2048+i)² per step, ~10⁸ index-pairs over 128 steps; the fix costs ~(2048+i) per step, ~10⁵ total; the ratio is about the average sequence length (around two thousand). Projections are wasted the same way: the bug QKV-projects S tokens per step, the fix projects one. A training forward looks like prefill: every position emits logits `[B, S, V]` and a one-token-shifted cross-entropy—do not draw the training graph and the decode graph as the same picture. A common modern block is RMSNorm → causal attention → residual → RMSNorm → SwiGLU FFN → residual, then a final norm and LM head, sampling only the last position.

**Follow-ups:** Under GQA, do you store cache at `H_kv` or `H`, and how are K/V broadcast to query heads; when stretching 2k→8k, does prefill’s quadratic term or decode’s read bandwidth blow up first; how a minimal implementation checks that every layer’s in/out stays `[B, seq, D]`.

**Extended questions:**
- If you swap Pre-Norm for Post-Norm, where do the forward dataflow and deep training stability diverge?
- Why cache K and V but not the attention output—which step still needs the current Q?
- When the prompt is unchanged and the user only appends one sentence, which layers’ caches can grow incrementally, and which must be invalidated?

**Reasoning:** Write down that each step’s new information is one token, so you only append K/V and must not rebuild Q over the full sequence; then sum O(n²) vs O(n) over 2k+128 to get a factor of about n. Accept by comparing per-token logits with the cache on and off.

## ARCH-030. A user message triggers a tool call: from the gateway to the streamed final answer, which steps cannot be skipped, and which two waits usually explode latency?

**Answer:** You generally cannot skip: auth and quota, assembling system instructions / history / tool schema into one request, input safety, routing to a concrete model, tokenizing with that model’s tokenizer, prefill, decoding with streamed tokens, parsing the tool call, executing it in a sandbox, appending the result and inferring again, then output filtering plus audit and billing. Time-to-first-token (TTFT) usually dies in the queue or in prefill of a huge history. The tool round-trip usually dies on the external API wait, or on a second prefill of the now-longer context. The loop is “model selects a tool → platform runs it → model reads the result → continues writing,” not a single forward pass.

**Deep dive:** Prefill runs attention over the whole prompt in parallel and writes KV; cost grows roughly quadratically with prompt length and is often compute-bound. Decode admits one new token per step and rereads KV, so it is often memory-bandwidth-bound. Streaming can push tokens after the first decode step, so TTFT ≈ queue + tokenize + prefill + first decode—do not confuse it with full generation time. Tool calls also need JSON/schema checks, timeouts, retries, and permissions; failures must be fed back to the model or degraded, not hung. In production, one user send may cross a gateway, a policy service, the inference engine, and a tool worker—about 1–4 extra hops. If history is concatenated raw every turn with no truncation or summary, prefill grows linearly in turns and attention gets quadratically more expensive.

**Follow-ups:** When a tool result is huge, how much do second-prefill and billing differ if you summarize vs stuffing the raw dump; should the output filter interrupt a stream mid-flight or wait for the full text, and what UX each choice hits; how to split TTFT in a trace into queue, prefill, first token, and tool RTT.

**Extended questions:**
- When the same user sends two messages in a row, under what conditions can the KV cache be reused, and when must it be dropped?
- How much context does the tool schema eat in the system prompt, and how do you hide tools the caller is not allowed to see?
- If the model emits a malformed tool call, do you retry, attempt a repair, or fail closed—and why?

**Reasoning:** Draw one user-visible turn as a state machine with a tool loop and mark the synchronous waits. Spend latency work only on the longest waits (queue / prefill / external RTT), not on micro-optimizing the sampling loop. Verify with span traces and timeout budgets, not an end-to-end average alone.

## ARCH-031. You only have a 32-query-head MHA checkpoint and need 8 shared KV groups. After mean-pooling K/V inside each group, can you serve it as GQA immediately? How much continued training restores quality, versus collapsing all the way to a single KV head — which tasks drop first, and where does tensor-parallel head sharding get stuck?

**Answer:** Mean-pooling the 32 KV heads in groups of 4 (or a single 1×1 mix) yields a legal 8-head K/V shape, but the query and output projections were trained against 32 independent key spaces. q_h·k_h becomes q_h·k_{group(h)} at once; softmax and the weighted V no longer match. The model runs, but it is not numerically equivalent. You need a continued-training stage (often freeze most layers, use a smaller learning rate on Q/O and the new KV) so queries re-align to the shared keys. Shipping the converted weights with zero extra training jumps short-text next-token loss and breaks long-range copy first. Eight groups usually keep coreference and needle-style retrieval better than one group (MQA). MQA saves more bandwidth, but a single KV head does not divide evenly across 8-way head-parallel tensor parallel — you have to replicate or reshard the hidden dim, and the communication pattern changes.

**Deep dive:** Shape-wise W_K and W_V shrink from `[D, 32·d_h]` to `[D, 8·d_h]`: view as `[D, 8, 4, d_h]` and average the size-4 axis. At inference the KV index is `h_q // 4`; the kernel broadcasts or gathers by group and does not change query-head width. Continued training has to restore the query–shared-key pairing, not relearn a language-modeling prior. Keep the original context length at first, mix the original pretrain/instruction data, and use a few thousand to a few tens of thousands of steps as an order-of-magnitude range, stopping on validation loss — not by copying a release recipe. The quality ladder is roughly full MHA ≥ grouped ≫ single KV head: multiple heads exist to capture different alignments in parallel; eight key spaces keep some specialization, one key space makes every query compete, so rare-span copy, multi-entity coreference, and long retrieval drop first. Head-sharded tensor parallel divides 32 KV heads across many ranks; 8 KV heads still map one-per-rank at TP=8, but going higher or down to 1 head forces K/V replication plus all-gather, or a switch to sequence/hidden sharding. “The shapes load” is not “the behavior equals a model trained as GQA.”

**Follow-ups:** How do mean-pool, keep-the-first-head, and a learned 4→1 mix differ in how many continued-training steps you need; with zero continued training, what breaks first — short-text cross-entropy or needle retrieval; if serving already shards heads at TP=8, why you still may refuse MQA even when bandwidth is tempting.

**Extended questions:**
- Write the reshape+mean of W_K from `[D, 32·d_h]` to `[D, 8·d_h]`, and the map from query head h to a KV head.
- During continued training, is freezing KV and training only Q and W_O safer for the already-learned value space than taking small steps on all four?
- If the target parallel degree is 8, sketch whether K/V are sharded or replicated for GQA-8 versus MQA, and whether attention is preceded by an all-gather.

**Reasoning:** The goal is a smaller KV without training from scratch; the constraint is an MHA checkpoint and head-sharded tensor parallel at serve time. Mean-pooling only fixes shapes; the Q–K pairing is a trained correspondence and needs continued training. Then compare 8 groups versus 1 group by how many independent key spaces remain, which tells you which alignments break. Finally ask whether H_kv divides the parallel degree before you collapse all the way to MQA. Validate on needle retrieval and coreference before versus after the continued-training run, and on whether each rank owns an integer number of KV heads.

## ARCH-032. Streaming decode keeps only the last 4k KV; once the window slides past the first tokens, perplexity spikes. Why does permanently pinning those few positions stabilize it?

**Answer:** Softmax forces each query’s weights to sum to 1. During pretraining the model learns to dump leftover mass on the first few tokens, which the causal mask makes visible to every later position — they become an attention reservoir. If you keep only the recent window, the ring evicts that reservoir, mass has nowhere legal to go, the distribution collapses, and perplexity jumps. Permanently keeping a few sink KVs plus the last W restores numerical stability. Evicted middle spans are still gone; this does not lengthen usable context.

**Deep dive:** The start of the sequence becomes a sink because BOS / the first tokens are visible to every later position from the first training step, so the model uses them as default slots. A streaming cache therefore splits in two: a fixed prefix (for example 4 sinks — an illustrative count) plus a sliding segment of W, dropping the middle. Another design adds a learnable sink logit in the softmax denominator so weights may sum to less than 1, and you may not even store those tokens. The boundary is sharp: sinks fix “is the distribution well-defined,” not “are facts outside the window still there.” Attention heatmaps normally show dark early columns; after sink eviction the heat wanders and next-token entropy rises. Do not read “can generate forever” as “remembers the whole conversation.”

**Follow-ups:** Why is the sequence start a more natural sink than some position in the middle of the window? What do learnable sink logits versus pinned first tokens cost in KV size and implementation? If sinks and the recent window are not contiguous in position id, do RoPE phases need remapping?

**Extended questions:**
- If you keep only the recent window and drop BOS on purpose, which failure shows up first on the generation curve — loops, garbage, or a sudden topic break?
- As you grow sinks from 1 to a few dozen, how do you trade stability against wasted attention mass?
- How would you use per-layer attention maps to show that a head depends on the sink rather than on recent semantic tokens?

**Reasoning:** The constraint is softmax normalization: mass must land somewhere. Explain why training puts the default slot at the start, compare “slot evicted” versus “slot pinned” cache layouts, and keep “generation does not blow up” separate from “middle content is still there.” Validate on a length–perplexity curve that kinks when BOS is evicted, not merely on whether tokens keep coming.

## ARCH-033. After turning on FlashAttention, forward activations drop by about half but the backward step gets slower and the peak memory comes back — which matrix does the tiled kernel recompute in the backward pass, and at what lengths does that recompute lose?

**Answer:** Tiled attention deliberately does not keep the n×n score matrix S or the softmax weights P. The forward pass stores Q, K, V, the output O, and per-row max and sum-exp (or an equivalent logsumexp). The backward pass needs P for dV = PᵀdO and for the softmax Jacobian, so each SRAM tile must rebuild S_ij from Q and K, restore P_ij from the saved row statistics, and only then accumulate into dQ/dK/dV. You drop O(n²) activations and buy roughly another attention-sized GEMM. When the sequence is only a few hundred tokens and n×n already fits on-chip cache, tile launch plus recompute can be slower than materialize-then-backward, and the peak can rebound because dO, O, and the recompute buffers are live together.

**Deep dive:** A standard identity for attention backward is dP_ij = dO_i·V_j, and the row reduction D_i = Σ_j P_ij dP_ij equals dO_i·O_i, so dS_ij = P_ij((dO_i·V_j) − (dO_i·O_i)), then dQ = dS K, dK = dSᵀ Q, dV = Pᵀ dO. Without P those formulas cannot be written, so the fused kernel uses the same (B_r, B_c) tiles as the forward pass, replays S_ij = Q_i K_jᵀ on-chip, and sets P_ij = exp(S_ij − m_i)/ℓ_i. Causal masks, padding, and dropout must follow the same in-tile rules as the forward pass; dropout also has to save or replay the RNG, or P will not match. Numerically this differs from materialize-then-backward only by floating-point associativity (ulp-level), and the arithmetic remains Θ(n²d). The losing regime is concrete: for example n=256, head_dim=64, FP16, one head’s S is only about 128 KiB and fits in L2, so recompute and indirect addressing cost more than the HBM you save. Decode with q_len=1 never had a fat activation and wants a decode kernel, not the training tiled forward/backward. When head_dim is so large that one SRAM tile holds only one or two query rows, occupancy drops, recompute count rises, and wall time looks bad. If an outer activation-checkpoint already recomputes the whole attention block, letting the kernel recompute P as well is double recompute — keep only one layer.

**Follow-ups:** Why the backward pass must keep both O and the row statistics, not just Q, K, V; how a causal mask written in the tile loop versus on a materialized S silently misaligns if one tile is missed; if selective checkpointing already wraps the layer, should the attention kernel still recompute P itself.

**Extended questions:**
- Sketch the tensors live in SRAM for one backward tile: Q_i, K_j, V_j, O_i, dO_i, and the dQ/dK/dV accumulators, and mark which must be written back to HBM.
- For the same batch and head count, is n=512 or n=8192 more likely to lose to a naive “materialize S” path, and is the bottleneck launch overhead or n² activations?
- If dropout attention discarded the mask bitmap in the forward pass, how do you guarantee the same positions are zeroed during backward recompute, and is the cost paid in memory or in RNG replay?

**Reasoning:** The goal is to avoid materializing n×n activations in the forward pass while still getting dQ/dK/dV that match the softmax Jacobian. The constraint is that both dS and dV depend on P explicitly, so you keep only the row statistics and O that can restore P, and recompute S/P inside tiles. Then ask whether the n² tensor would have fit anyway: at short n or q_len=1 the traffic you save is near zero and only tiling overhead remains. Check that the backward peak no longer holds an [n,n] workspace, and where the short- versus long-sequence wall-time curves cross.

## ARCH-034. If you turn on FlashAttention, does dense attention's arithmetic cost become linear in sequence length?
**Answer:** No. It uses tiling and online softmax to avoid materializing the full attention matrix, reducing memory traffic and intermediate storage. The arithmetic work of standard dense attention still grows approximately quadratically with sequence length.
**Deep dive:** Distinguish peak memory, HBM traffic, arithmetic complexity, and measured runtime.
**Follow-ups:** Why do memory and time not necessarily scale in lockstep when length doubles? Intermediate storage, arithmetic work, and memory access are different resources, and a given implementation also has fixed overhead and changing hardware utilization.
**Extended questions:**
- When comparing ordinary attention with FlashAttention, how do you keep the mask, precision, and input shapes the same?
- Why does "not materializing the full attention matrix" not mean "not computing the relationships among all visible token pairs"?
**Reasoning:** First identify the bottleneck resource, then decide what I/O optimizations can and cannot solve.

## ARCH-035. When both activate about 50B parameters, why is decode latency for a 400B-total sparse model often worse than a 50B dense model?

**Answer:** A decode step is usually memory-bandwidth bound: kernels must read the weights and KV used this step. A 50B dense model reads about 50B contiguous parameters. A MoE with, for example, 400B total and ~50B active has similar paper FLOPs but still gathers chosen experts from scattered addresses and pays cross-device all-to-all. At small batch, communication dominates and devices holding idle experts wait on a sync, so wall-clock latency often exceeds a 50B dense model with mature kernels and no router.

**Deep dive:** Dense: two or three large GEMMs per layer, regular shapes, high utilization, latency that scales smoothly with batch. MoE: routing is discrete, different tokens in the same step take different experts, and one large kernel splits into several small ones. Holding 400B weights forces expert parallelism or offload—the first adds communication, the second adds host-bus round trips. Attention is still dense on both sides, so KV cache can be the same size; sparsity does not automatically shrink cache. When it wins: large-batch training or prefill is compute-bound, and sparsity can buy more capacity at similar FLOPs. When it loses: online decode, tiny batch, tight latency SLAs—dense is more predictable. If decode batch is large, expert locality is good, and the interconnect is fat, a MoE can match or beat same-active dense. That is a condition, not the default. “Sparse” does not mean “fewer weight reads, therefore faster.”

**Follow-ups:** How does MoE versus dense look in prefill compared with decode? After experts are offloaded to CPU, what does latency look like on a cold-expert hit? At matched quality, which metrics decide 50B dense versus a 400B-class MoE?

**Extended questions:**
- Sketch tokens/s versus decode batch from 1 to 64 and mark where MoE and dense cross.
- If sparsity comes from weight pruning rather than expert routing, do you still pay a communication tax at decode?
- Why does a multi-tenant mix of lengths and domains worsen MoE load imbalance?

**Reasoning:** The goal is low decode latency. The constraints are bandwidth, communication, and kernel efficiency—not paper FLOPs. Compare “read 50B contiguous weights” with “route + gather experts + sync.” Check with batch-1 time per token and a communication timeline.

## ARCH-036. With sliding-window attention at W=4096 on a ~20k-token contract, how far can layer 4 versus layer 32 still “reach,” and how much KV can you drop?

**Answer:** Sliding-window attention lets position i attend only to [i−W+1, i], so one layer is O(n·W) rather than O(n²) and the KV cache is a ring of the last W slots. Information can hop across layers: layer ℓ has a theoretical receptive field of about ℓ·W, so for example W=4096 gives ~16k tokens at layer 4 and ~131k at layer 32 — a 20k-token contract is “on paper” inside the deep field. That hop is a stack of weighted mixes, not a copy of the distant clause, so exact retrieval of a disclaimer still often fails. KV storage falls from n to W; at n=20k and W=4096 you keep roughly one fifth of a full cache.

**Deep dive:** Each layer keeps a K/V ring of length W: the new token writes at `head % W` and overwrites the expired slot, so you never shift the whole buffer. Attention FLOPs scale like n·W·d instead of n²·d. Memory is about 2 × layers × KV heads × head_dim × sequence × bytes — for example 32 layers, 32 heads, head_dim=128, FP16, a 20k full-attention cache is on the order of several GB and shrinks by about W/n under the window. The ℓ·W field only guarantees a dependency path; softmax in the middle layers smears a far signal, so needle-style facts (numbers, carve-outs) die outside the window. In practice people alternate window layers with a few global layers, or pin the first few sink tokens: global layers carry titles/tables of contents, window layers carry local wording. Do not treat theoretical field size as an effective retrieval radius.

**Follow-ups:** How do sliding windows differ from block-sparse and linear attention in visibility and KV layout? Why is stacking more window layers not enough, so you still insert global layers? If the window slides and you do not keep sinks, how does the attention distribution break? As W goes from 1k to 8k, how do quality, memory, and decode latency each move?

**Extended questions:**
- Given depth L, window W, and token distance Δ, how do you tell “theoretically reachable” from “retrieval will fail”?
- With a ring KV plus attention sinks, which cache spans are pinned and which are overwritten?
- If the job is to check two clauses for contradiction, how should the window share work with retrieval or global layers instead of only raising W?

**Reasoning:** The goal is KV memory; the constraint is that a distant clause must still be citable. From “each layer sees only W” you get the complexity drop and cache n→W; stacking gives field ℓ·W, which you compare to 20k versus 16k/131k. Then separate mix-from-copy so theoretical radius ≠ usable retrieval, and estimate savings as W/n. If acceptance is verbatim match, do not stop at the field-size number.

## ARCH-037. You need to send a product photo plus “what kind of sole is this?” into a decoder-only chat model. Do you concatenate patch tokens into the same self-attention stream or inject them with cross-attention, and how do the two paths differ for KV cache and localization?

**Answer:** Both work; cache and interaction differ. Concatenation: a vision encoder or linear map turns patches into visual tokens that share causal self-attention with text. Simple to implement, but every visual token occupies KV, so prefill and cache grow linearly with resolution. Cross-attention: language queries read image K/V; the image side is computed once and reused every decode step and does not consume the text sequence’s length quota. Alignment is heavier, and pointing at the sole depends on resolution and how dense that cross-attention is.

**Deep dive:** A vision Transformer tiles the image into regular patches (for example 16×16 pixels), adds position encodings, and treats them as a sequence; a chat model still needs a projection or compressor into the language hidden size. Early fusion lets text tokens see every patch, which helps “what material is this region,” but hundreds to thousands of visual tokens eat the context window. Cross-attention treats the image as a second source sequence; the score matrix is [text length, vision length] and source K/V can be cached. Either way the model learns patch–text co-occurrence, not human seeing; tiny OCR, fine materials, and unusual lighting still fail often. Evaluate with questions that point at a region, not “is there a shoe?”

**Follow-ups:** When you raise input resolution, how does KV growth compare for concat vs cross-attention? What spatial detail do you lose by compressing to a few dozen visual tokens? On text-only turns, do you still keep empty visual placeholders?

**Extended questions:**
- Estimate how many visual tokens you get at side length about 336 with patch size 14, and what fraction of a 4K window that is.
- When the user says “that bit at the bottom left,” how does each fusion path get spatial coordinates into the model?
- How should visual tokens share work with a tool that crops a higher-resolution patch?

**Reasoning:** Window and KV budget decide how many visual tokens you can afford; whether you need fine localization decides whether you raise resolution or crop again. Concat is simple and spends window; cross-attention saves sequence length and costs engineering. Choose with localization items plus time-to-first-token and KV occupancy, not with an abstract “which understands images better.”

## ARCH-038. An 8-expert top-2 MoE grows total parameters several times. Why don’t per-token FLOPs grow by the same factor, and why can it still be slower online?

**Answer:** MoE usually replaces only the FFN with E experts; attention stays shared. Top-2 means each token runs two expert FFNs. Versus a same-width dense block, FFN parameters are about ×8 and total model parameters often land around 4–6× (attention is not copied), while FFN FLOPs are only about ×2, so per-token FLOPs rise by a slice after you add attention. Online slowness is usually not paper FLOPs: hot experts overload, all-to-all ships tokens to the devices that hold those experts, and small-batch expert GEMMs get shredded.

**Deep dive:** Data flow: hidden → router scores over E → pick top-k and gates → send tokens to those experts → weighted sum back. A capacity factor > 1 lets an expert take extra tokens; overflow is dropped or spilled, and drops hurt quality. Load balance uses an auxiliary loss on expert-frequency variance or a learned bias on routing logits so winners do not keep winning. With expert parallelism, each forward pays at least one all-to-all whose latency tracks routed tokens and inter-node bandwidth. At decode batch = 1, 2 of 8 experts run and the other devices idle; theoretical FLOP savings die in a sync. Edge cases: total parameters set memory (experts typically all resident); activated parameters set FLOPs. Do not estimate latency of “hundreds of billions total / tens of billions active” as if it were a tens-of-billions dense model. Training must also stop expert collapse, where a few experts eat almost every token and capacity decays into a slightly wider dense FFN.

**Follow-ups:** How does load look when tokens choose experts versus experts choose tokens? Why is MoE especially expensive at decode batch = 1? How do you balance load without an auxiliary loss?

**Extended questions:**
- If you change top-2 to top-1, how do FLOPs, quality, and collapse risk move?
- Compared with doubling a dense FFN, what expressivity do you lose by sending the same hidden state through two experts and gating them?
- How do you monitor per-expert token share and set an alert threshold?

**Reasoning:** Split the bill into total parameters, activated parameters, communication, and balance. Plug in E = 8, k = 2 to get “parameters up several times, FFN FLOPs ~×2,” then explain how all-to-all and idle devices inflate wall time. Confirm with expert-load histograms and the communication time share.

## ARCH-039. Moving a MoE serving window from 8K to the 100K class, does full attention blow compute or KV first? How does hybrid compression across layers help?

**Answer:** Prefill full-attention FLOPs grow with length squared: 8K → 100K is about (100/8)² ≈ 156× attention compute. KV cache grows linearly, about 12.5×. At a few thousand tokens you often feel KV crowding out concurrency first; at 100K, prefill wall time and KV blow together. MoE sparsifies FFN activation, not those two terms. Hybrid layers help: some layers use windowed or block-retrieved sparse attention (fewer KV entries and dots), some run compressed dense global mixing, and you keep a sink at the start plus a recent window so long-range precision is not deleted entirely.

**Deep dive:** Standard multi-head / grouped attention stores one K, V per token per layer and dots against all history. Using 40 layers, 8 KV heads, d_h = 128, BF16 as an illustration: ~160 KiB/token, ~16 GiB per 100K sequence—unrealistic once you multiply by concurrency. Prefill attention scales like L·n_q·n²·d_h; a 12.5× longer n multiplies the quadratic term by ~156×, while FFNs only grow 12.5×, so attention dominates again. Architecture knobs (usually trained in): (1) sliding window—keep W KV slots, compute O(nW); (2) block-sparse or compressed indexes—fold every m tokens into a retrievable entry, score cheaply, attend only to top-k blocks; (3) latent KV—cache a low-d vector and expand when used; (4) alternate layer types so not every layer pays full n². Edge cases: unselected tokens vanish, so “find the sentence at 80K” degrades; stacking layers only expands receptive field indirectly. MoE routing picks FFNs per token and is orthogonal to attention compression.

**Follow-ups:** Why can’t FFN expert sparsity save KV? After 40 sliding-window layers, what is the theoretical receptive field, and why is actual retrieval still weak? Do latent KV and grouped attention shrink the same dimension?

**Extended questions:**
- If even layers stay full attention and odd layers are windowed, how do you roughly trade KV against long-range accuracy?
- If a 100K prefill must finish in 10 seconds, do you cut the n² term or the expert FFN?
- How would you use a precise-lookup set and a global-summary set to check that sparse layers were not cut too far?

**Reasoning:** Apply the 8K → 100K scale factor separately to O(n²) attention and O(n) KV/FFN to see what blows first. MoE only sparsifies FFNs, so attention structure has to change. Assign layer types to “keep global / keep local / keep sink,” then accept with lookup tasks and summary tasks as a pair.

## ARCH-040. On a multi-step calculation, a reasoning model spends an order of magnitude more latency writing intermediate steps than an ordinary instruction model. Which tasks deserve that test-time compute, and how do you cap the thinking budget?

**Answer:** Spend it where intermediate steps are checkable and a single error is expensive: multi-step calculation, code that must pass tests, multi-constraint planning. Those models are post-trained with verifiable rewards to emit inspectable long reasoning, then spend more tokens at inference for accuracy—test-time compute. Chitchat, closed-set classification, and facts already sitting in retrieved text should not take this path: they get slower and costlier and may wander. Cap with a max thinking-token count, a timeout, or an effort tier, and route easy items to an ordinary instruction model.

**Deep dive:** “Think step by step” in a prompt only changes the output format. A reasoning model puts think-then-answer into the training objective and learns to self-check and switch approaches from signals such as answer correctness or tests passing. The cost is output growing from tens of tokens to hundreds or thousands, so wait time tracks thinking length. Easy items may overthink; the written chain is generated text, not a guaranteed trace of internal computation, and showing it to users can mislead or leak. Log thinking tokens against correctness and plot accuracy versus cost to find diminishing returns. Bucket eval by difficulty so easy-item averages cannot hide the curve.

**Follow-ups:** How do failure modes differ between prompted step-by-step thinking and a model trained to reason? If thinking tokens are billed separately, how do you explain the wait to a user? When the budget is exhausted, do you cut thinking and answer anyway, or fail and retry?

**Extended questions:**
- What routing rule sends a support ticket to a reasoning model without also sending “where is my tracking number”?
- Why might a reasoning trace be shown to a reviewer but not to the end user?
- How do you tell that adding more thinking tokens has entered diminishing returns?

**Reasoning:** Latency and unit cost are the constraints; hard-item accuracy is the goal. Ask whether correctness is automatically checkable and whether failure is expensive; only then buy test-time compute. Routing plus a hard budget stops every request from thinking deeply. Accept with bucketed accuracy and that class’s share of total cost.

## ARCH-041. A function is missing its middle, and both sides are already written. Why might iteratively unmasking tokens be less awkward than left-to-right continuation? What do you do if the length estimate is wrong?

**Answer:** Left-to-right generation appends at the end. Filling a hole usually means dropping the suffix and rewriting, or training a special prefix-suffix-middle format. Masked diffusion treats each output site as a slot that can be denoised in parallel: known tokens stay, unknown ones start as masks, each step predicts every remaining mask, keeps high-confidence tokens, and continues on the rest. Left and right context condition the same sequence; the suffix does not have to be destroyed first.

**Deep dive:** The forward process gradually replaces discrete tokens with masks (or moves them into an absorbing state). The reverse model learns “given the currently visible sequence, what are the masked sites.” Inference step count T is its own knob: small T is fast and coarse, large T refines—unlike autoregression, where steps equal output length. The cost is that you must pick or estimate a length L and draw L slots; too short truncates, too long fills holes or end markers. In practice you estimate L with a heuristic or a small model, or score several lengths in parallel. A step-by-step KV cache does not transfer cleanly, because each step rewrites the whole span rather than one new position. On long-range coherence and multi-step reasoning these models still trail a mature autoregressive stack. One-shot masked fill-in is a single step, not an iterative full-span loop, and should not be conflated with this denoising cycle.

**Follow-ups:** How do you plot a step–quality curve and pick T under a latency budget? Why leave low-confidence sites for later unmasking? Can length be a variable absorbing process instead of a fixed L? How do you turn indentation and bracket balance into hard constraints during denoising?

**Extended questions:**
- Compare prefix-suffix-middle autoregressive infill with masked-diffusion infill: where do implementation complexity and cache utilization differ?
- If the user later edits only three middle lines, can a diffusion model denoise just those lines? What would an autoregressive model have to do to match that?
- If training predicts all masked sites independently versus jointly, what happens to repeated tokens and global consistency?

**Reasoning:** The known set is context on both sides; the unknown set is the middle span. A process that can unmask an arbitrary subset matches that partition; strict left-to-right treats an already-written suffix as unknown. Slot length is an extra discrete degree of freedom and must be estimated or searched on its own.

## ARCH-042. A user drops in a ~2M-token repository and one security-audit question; the model window is 128k. Why have the model write probing code that reads files on demand, instead of chunking first and retrieving?

**Answer:** Chunk-then-retrieve assumes you already know which fragments the audit trail will land in. Multi-hop questions such as “who uses this key, then follow the data flow” break embedding recall. A recursive setup leaves the repo in the environment (tree, read-only file API). The main model gets a short instruction and a few primitives: list a directory, read a span, spawn a child call. It decides what to cut and what to ask each slice, then reduces child results; children can cut again.

**Deep dive:** You are dodging three things at once: a hard window cap, ignored middle spans in very long context, and the cost of stuffing the whole repo into the prompt. This is a cousin of ordinary tool use—the model chooses an action, the environment runs it—except the raw repo does not enter attention until a slice is “seen.” Unlike retrieval, the exploration policy is chosen at generation time, not at index time. The cost is tens of model calls per user question, higher latency and spend; child errors roll up; probing code must run in a read-only, no-network sandbox with timeouts and a call cap. Single-file questions far under the window are not worth this—just put them in context. Coverage is a first-class output: unread directories must be declared; the system must not pretend the audit finished.

**Follow-ups:** How do you cap recursion depth and total token budget so the bill cannot explode? If a child returns an obvious contradiction, what lets the parent notice? If the repo contains key files, how do you isolate the sandbox and the prompt? How is this different from a map-reduce summary tree followed by one final answer?

**Extended questions:**
- Design hard limits: max child calls, max characters read per level, wall-clock time; on overflow, how do you degrade to “report only what was seen, plus coverage”?
- On “compare two protocol implementations across thirty files,” what does on-demand probing miss versus a prebuilt call graph?
- How do you log each child’s inputs and outputs so a missed path can be reconstructed later?

**Reasoning:** The constraint is that two million tokens cannot enter attention at once; the goal is to spend reading budget according to the question. The minimal mechanism keeps the control plane in the prompt, the data plane in the environment, and turns reading into on-demand I/O via recursive child calls. Without a coverage output, the system systematically under-reports what it never saw.

## ARCH-043. Tickets must be routed into 20 fixed categories. When should you score a closed label set with a classifier instead of letting an LLM generate the class name, and how do you use calibrated probabilities to auto-handle versus hand off?

**Answer:** When the output must land in a closed set, latency must be stable, and you need a thresholdable confidence, score fixed labels with a classifier rather than letting a generator emit a class name: you do not get typos or extra prose, one forward pass is enough, and probabilities can be calibrated on their own. Keep the generator for writing a reply or open-ended reasoning. Auto-handle only when the calibrated top-class probability clears a threshold you set on a validation set and the class is not high-risk; otherwise hand off to a human or escalate to a generator.

**Deep dive:** Generated class names fail as aliases, extra punctuation, or occasional refusals, which forces a second parser; with sampling on, the same ticket can jitter into a neighbor class. A closed classification head softmaxes a fixed-width logit vector, so parse cost is zero, but it can still pick the wrong class—valid format is not a correct label. Calibration (for example temperature scaling, checked with a reliability diagram and calibration error) is what makes a threshold meaningful: if the model says 0.9 it is right about 90% of the time, you can write “above θ, close automatically.” Classifiers do not write rationales and cannot handle a class that did not exist in training; a new line of business needs a new label and a re-eval. A common layout is classifier for routing and guardrails, generator only when text must be written. Evaluate closed-set accuracy, calibration error, p95 latency, and the generative baseline’s parse-fail rate (usually not zero).

**Follow-ups:** Why does “emit JSON” still produce type failures while a closed classification head does not? What is dangerous about an uncalibrated 0.99? When a 21st class appears, what happens if you keep the old classifier?

**Extended questions:**
- Draw a reliability diagram on the validation set and show how you would choose the auto-close threshold.
- Which tasks should not use a closed classification head (must quote the source, must call tools in several steps, labels unknown in advance)?
- In a classifier-plus-generator cascade, how do you stop the two models from disagreeing on what “urgent” means?

**Reasoning:** Read the output contract: closed decision vs open text. Closed, thresholded, stable-latency work belongs on a scoring model; sentence writing belongs on a generator. Thresholds must come from calibrated probabilities, or automation has no honest coverage–risk trade-off. Accept with calibration error, wrongful auto-close rate, and latency.

## ARCH-044. Five million intent-classification calls per day are too expensive on a 70B. What size of small model should take the first hop, and which online signals should escalate a request?

**Answer:** Closed-set decisions—intent, slot fill, a first safety screen—can usually be taken first by a task-finetuned small model of a few hundred million to about a few billion parameters, because the label set is finite and you do not need open-ended generation. Compare quality, p95 latency, and cost per call on your own eval, not a general chat leaderboard. Write escalation as pre-agreed uncertainty: high classification entropy, failed slot validation, the user contradicting themselves, or a high-risk intent—then send the raw text to the large model. Do not escalate at random.

**Deep dive:** “Small” is not a legal threshold. In practice it often means a model that finishes in tens to a couple of hundred milliseconds on a single consumer accelerator or on-device chip, from a few hundred million to low-tens-of-billions of parameters; some teams still treat a 7B-class model as the small hop. Cleaner data, distillation, and longer training let them approach a large model on a narrow task; world knowledge, multi-step reasoning, and long-tail phrasing stay weak. The system is a cascade: high-confidence traffic returns from the small model, the rest escalates. On-device you also quantize, pick a runtime, and check tokenizer coverage for the task language. Eval must include tail intents and adversarial paraphrases so head classes cannot fake “good enough.” What share of traffic stays on the small model comes from your calibrated threshold curve, not a fixed percentage.

**Follow-ups:** How do you calibrate the small model’s probabilities before using them as an escalate threshold? Can distillation copy the teacher’s false refusals? When should you grow the small model instead of cascading forever?

**Extended questions:**
- Design “small model first, escalate on low confidence” routing and list four log fields you would persist.
- What can 4-bit on-device quantization do to classification calibration?
- How do you show a finetuned small model can replace the large model on most traffic, rather than overfitting the dev set?

**Reasoning:** The goal is closed-set coverage under latency and unit-cost caps. Confirm the task is truly closed and price a mistake. If so, a small model plus a threshold cascade is the default; open generation and heavy reasoning stay on the large model. Validate with task quality, latency, escalate rate, and how often escalation saves the request—not with parameter count.

## ARCH-045. The team lets an agent edit its own prompts and tool wrappers, using public-benchmark score as the overnight-loop signal. How do you make sure it is improving capability rather than the evaluator?

**Answer:** Self-improvement feedback is real capability only when the scoring rule is harder to rewrite than the system under test. The harness, gold labels, and a hidden holdout must live outside the agent’s filesystem and permissions: it may submit artifacts, not edit the scorer or read hidden items. Keep two tracks—a development set it can see and a holdout it cannot. Only a holdout rise counts as improvement.

**Deep dive:** Typical shapes include models training on their own synthetic data, self-play, prompt/tool edits checked by an executor, and another model used as a reward. The shared failures are score-hacking and mode collapse (synthetic text getting more homogeneous, entropy falling). Write isolation as mechanism: scorer read-only and hash-pinned; sandbox offline and unable to write the eval directory; every edit leaves a diff and a trace; stop for a human if score or tool-call count jumps past a threshold. A public-set spike with a flat or falling holdout is treated as teaching to the test. If generator and evaluator share weights or a writable prompt, the evaluator is contaminated—split models or at least freeze the eval side.

**Follow-ups:** How does this differ from automatic hyperparameter search (does the search space include the evaluator itself)? When should synthetic self-training watch diversity instead of loss alone? If the evaluator is also a model, how do you stop the pair from inflating each other’s scores? How large a score jump is a tripwire?

**Extended questions:**
- Split the evaluator for a code-optimization loop: which of correctness, runtime, and readability can be fully automatic, and which need human spot-checks?
- How do you detect mode collapse in a synthetic-data loop (repetition, embedding clumping, downstream drops)?
- List signals that must go to a human: holdout drop, attempts to read the eval directory, output-length collapse, abnormal tool-call counts.

**Reasoning:** The sign of the feedback is defined by the evaluator. Freeze “what better means” and put that definition out of reach of the thing being optimized, then allow self-edits. Any loop that can edit the evaluator or that only optimizes a visible proxy score is treated as hacking, not capability gain.
