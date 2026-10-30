# Decoding, Context and Long Context

[← Back to contents](../README.md)

## GEN-001. A 13B legal assistant starts padding mid-answer with empty stock phrases. A 1.5B base with the **same tokenizer**, not domain-aligned, can emit logits each step. Contrastive decoding subtracts the amateur scores from the expert’s. How do you suppress phrases both models love without deleting statute numbers and party names, and how do you accept α and the candidate cutoff?

**Answer:** Run the 13B (expert) and 1.5B (amateur) every step and, on **aligned vocab ids**, set `z′ = (1+α) z_exp − α z_ama`, then softmax only on tokens whose expert probability is at least `β · max p_exp`. Tokens both models score high (stock phrases) are pushed down; tokens only the expert scores high (article ids, docket numbers) tend to remain. Different tokenizers: do not subtract—misaligned dims are noise. Too-large α also kills rare proper names; too-small β lets the amateur’s tail in.

**Deep dive:** The amateur should supply a fluent-but-empty prior, not a second domain expert: a small unaligned base ranks high-frequency glue like “under / applicable / provisions” high, while a domain-SFT 13B still puts mass on concrete citations. After the subtraction, dimensions where the expert outranks the amateur rise. The head set V_head = {v | p_exp(v) ≥ β p_exp,max} stops the amateur from scoring ids the expert almost never considers; try β around 0.1 (illustrative, sweep on your set). Implementation must align steps: same input ids, same position, then subtract; stop, temperature, and top-p apply to z′ afterward, same order as ordinary decode. Cost is an extra 1.5B forward per step: usually cheap next to 13B, still bandwidth, and at large batch you either fuse both forwards or accept higher TPOT. Failures: (1) amateur too weak or too close to the expert → contrast ≈ 0; (2) party names and internal document ids are almost unseen by the 1.5B, so a large α over- or under-scales expert-only spikes—score exact match on a gold name set, not only “fewer stock n-grams”; (3) legal repeats in code or JSON (brackets) look like “both models love them”—disable contrast on those routes or keep a never-penalize symbol table. Do not substitute a frequency penalty: that term follows **counts already in this sequence**; contrast follows the **difference of two model priors**, so it fires on the first occurrence of a stock phrase. Eval: stock n-gram rate, citation exact match, name recall, plus an off-contrast control; pick α, β at the knee of that pair of curves.

**Follow-ups:** The 1.5B is missing 4k special tokens the 13B has—refuse contrast, map to the intersection, or swap in a same-tokenizer small model; α=0.5 cuts stock phrases and starts breaking article ids—lower α or tighten β first; at temperature=0, should contrastive decoding still differ from greedy on the expert alone.

**Extended questions:**
- Write z′ and V_head, and show how β=0 lets the amateur dump mass on the expert’s near-zero dims.
- Using the position of the **first** stock phrase, how do you prove this is not a reskinned frequency penalty?
- If the 1.5B and 13B drift one step apart at high concurrency, what “sudden voice change” does that look like?

**Reasoning:** Suppress tokens both models like; keep identifiers only the expert ranks high. Logits must share a tokenizer, and the proper-name set is not stock phrasing. Linear contrast with a head cutoff; sweep α, β on stock-rate versus exact match together. Accept when names hold and stock falls, not when answers merely get shorter or less templated.

## GEN-002. Some decode steps have p_max=0.95; others need 20 tokens to reach mass 0.6. On which kind of step does fixed top-k=50 or top-p=0.9 admit a bad token or drop a good one?

**Answer:** On a sharp step (p_max=0.95), top-k=50 still admits 49 near-zero tokens, and one draw can jump off-distribution; top-p=0.9 often keeps one or two candidates and stays safe. On a flat step (20 tokens to reach 0.6), top-p=0.9 swallows a long tail and gets noisy, while top-k=50 may cut before legitimate synonyms enter. k is a hard window, p is a mass window; they fail on opposite shapes.

**Deep dive:** Sort probabilities descending. Top-k keeps k and renormalizes; top-p keeps the shortest prefix whose mass is ≥ p. Both usually sit after temperature. A common stack: divide by T, cap with top-k, then nucleus with top-p. Factual or code work wants a tight nucleus or greedy; open writing can use p around 0.9–0.95. Traps: p=1 is no cutoff; huge k is a no-op; a large k on a sharp step looks like nucleus sampling while drawing garbage. Debug with the distribution of candidate-set sizes, then score repetition, illegal tokens, and task accuracy together.

**Follow-ups:** When both k and p are set, which filter typically runs first; why chat decoding rarely uses k alone; how one step’s histogram tells you to tighten p versus k.

**Extended questions:**
- Sketch CDFs for a sharp and a wide distribution and mark the cut points for k=50 and p=0.9.
- For code completion, what order of p would you use, and with what stop condition?
- If you sample a vocab-tail junk token, how do you use logged candidate-set size to tell oversized k from oversized p?

**Reasoning:** Hold up a sharp histogram and a flat one, then see whether a fixed set or a fixed mass fails there. Tighten the nucleus on structured work; loosen it for open text but keep k as a cap. Candidate-set size is the debug signal, not a single magic number.

## GEN-003. Raising temperature from 0.2 to 1.2 makes JSON keys drift. What happened between logits and sampling, and why can temperature=0 still disagree across calls?

**Answer:** Temperature T divides logits before softmax: T<1 sharpens the distribution; T>1 flattens it so low-probability tokens get drawn more often. Moving 0.2→1.2 shrinks the odds ratio between the right key and a near-miss, so names and punctuation jitter. T=0 should pick the max logit, but batched kernels, nondeterministic reductions, and floating-point ties can change the winner, so T=0 is not bit-stable.

**Deep dive:** pᵢ = exp(zᵢ/T) / Σⱼ exp(zⱼ/T). A logit gap Δ becomes an odds ratio exp(Δ/T) instead of exp(Δ); 0.2→1.2 shrinks that gap by 6×. Structured tasks want low T plus a legal-token mask or grammar; creative writing is where you raise T. T changes sharpness only, not rank order—rank changes need bias or a mask. Too-high T lets a long tail break JSON; too-low T plus greedy repeats a wrong local mode. Tune T together with top-p, retries, and schema checks, and score stability by self-agreement on the same input.

**Follow-ups:** When T and top-p are both on, do you scale then truncate or the reverse; why some services document that T=0 is still not byte-identical; how you pick T from self-agreement and schema pass rate.

**Extended questions:**
- For logits [4.0, 3.5, 0.1], compute the top-two probability ratio at T=0.2 and T=1.2.
- On which tasks would you freeze T=0, disable nucleus sampling, and switch to constrained decoding?
- When JSON sporadically breaks in production, how do you attribute using logged T, top-p, and finish_reason?

**Reasoning:** The goal is stable fields. Softmax is scale-sensitive, so read how T changes odds, then subtract system nondeterminism at T=0. Accept with multi-call agreement and parse success, not a single “looks right.”

## GEN-004. Around token 800 the model starts looping the same closing sentence; temperature is 0.7 and there is no frequency penalty. Which side do you touch first, and which metrics prove you did not just change the skin of the loop?

**Answer:** Separate a short greedy-style loop from a high-probability phrase reinforcing itself under sampling. Immediately add frequency/presence penalties or a repeated n-gram ban, and check whether the prompt asks to “keep going” or “be thorough,” which forces self-continuation when there is no new information. If the loop becomes near-synonym repetition or copied list items, you only changed the skin—tighten stop conditions, cap length, or change decoder/model for long outputs. Track repetition rate, distinct-n, and longest repeated n-gram together; do not only ask whether that exact sentence remains.

**Deep dive:** Under autoregression, emitted tokens re-enter context; once a high-probability phrase appears, the conditional distribution raises it again and a stable cycle can form. Temperature 0.7 does not prevent this if on-cycle tokens still dominate the tail. Frequency penalty lowers logits of seen tokens and helps token-level loops; it is weaker against synonym cycles, where you want sequence-level detection (for example halt or down-weight a repeated 8-gram). If the pretraining mix contains footers or scrape noise, the base model itself recites; decoding penalties are an inference patch. Evaluate on a forced-long-output set with repetition-2/3 and distinct-2, plus a human check that you did not replace the loop with filler.

**Follow-ups:** Why does beam search produce bland repeated sentences more often in open-ended generation? What does frequency penalty punish versus presence penalty? Can you apply the same penalties inside code, where brackets and repeated identifiers are legal?

**Extended questions:**
- Given a looping token sequence, mark the step where the conditional distribution locks onto itself.
- Which legitimate repeated field names can a repetition penalty break in JSON extraction?
- How do you detect a loop in a stream and cut it without clipping intentional parallelism?

**Reasoning:** Loops are decoding dynamics plus a “write long” objective. Cheap decoding constraints stop the bleeding; n-gram metrics confirm it is not a reskinned loop. If it still recites under several penalties, blame the model and data and consider a different model or a shorter task. Regress both repetition and task completeness on a long-output set.

## GEN-005. For open-domain chat, JSON ticket extraction, and term translation, which decoding method among temperature sampling, greedy, top-p, beam, and top-k do you ban on each API, and what breakdown does that avoid?

**Answer:** Term translation: prefer beam or low-T greedy and disable high-T nucleus sampling so terminology does not jitter; beam fits sequences with a fairly tight correspondence. Open chat: disable pure greedy and large-beam search—greedy loops, large beams go safe and bland; use moderate T plus top-p. JSON extraction: disable high T, wide top-p, and beam (beam is bad at hard schemas and costs ~B); use T near 0 plus a schema/grammar mask, greedy if needed.

**Deep dive:** Greedy takes argmax each step: fast, closest to reproducible, fails via repeated n-grams and local optima. Beam keeps B partial hypotheses: fails via length bias, boilerplate, and low diversity, at about B× compute. Top-k is a fixed set: garbage on sharp steps, clipped synonyms on flat ones. Top-p is a mass cut: too-large p becomes a long noisy tail. Temperature only changes sharpness; high T amplifies every failure above. A typical order: penalties/bias → divide by T → legal mask → top-k/top-p → sample. T=0 can still disagree across calls from floating-point nondeterminism. Sweep parameters on the task set for accuracy, repetition, diversity, and parse rate.

**Follow-ups:** Besides bland text, which latency and objective mismatches keep beam out of chat; in which order k and p filter when both are on; how you sweep a held-out set without overfitting that set.

**Extended questions:**
- Characterize a prompt that drives greedy into “yes yes yes,” and give three mitigations: repetition penalty, a stop rule, and a T change.
- How do you measure schema pass rate for “T=0, unconstrained” versus “T=0 plus a regular grammar” on JSON?
- If beam has no length penalty, how does it systematically bias short reference translations?

**Reasoning:** Classify the three APIs by “is there a fairly unique target, is randomness allowed, is the format hard.” Map each decoder onto those failure modes. Ban strategies that break the constraint, then sweep the survivors on metrics.

## GEN-006. After a runtime swap the model keeps writing a fake User: turn and finish_reason is often length. In what order do you inspect the chat template, eos_token_id, and max_tokens?

**Answer:** Template first: training taught a specific end token at the close of the assistant turn. If the new runtime does not close in the same place, or leaks the next `User:` header into the model, it will never emit that end token and will impersonate the next turn. Then check that the runtime’s stop set includes that id—some models use a different id for eos versus end-of-turn. Only then look at max_tokens: if no end token is ever sampled, generation hits the cap and `finish_reason=length` means truncation, not a clean stop.

**Deep dive:** Stops come from three places: a trained end token the model emits; caller string stops; and the length cap. Chat models are extremely template-sensitive; a mismatched template is the leading cause of “won’t stop + speaks as the user.” The tokenizer eos id, the generate stop ids, and whether special tokens are stripped must agree. Applications should branch: `stop` is usable; `length` needs a completeness check, then a continuation, a larger budget, or a template fix. Fine-tunes that never placed an end token at sample ends can still drive loss down and then ramble forever at decode.

**Follow-ups:** Why some models must list both eos and the dialogue end token in the stop set; how you continue a `length` cut without repeating the already-streamed prefix; how finish_reason mix becomes a ship gate.

**Extended questions:**
- Design a minimal repro that only changes the template delimiter and predict finish_reason flipping from stop to length.
- With several end tokens, should the stop set and the returned text include the token that fired?
- If fine-tune data omits the end token, can training loss still fall, and what do you see at inference?

**Reasoning:** Stopping requires a detectable end signal. Align the trained signal (template + id) first, confirm the runtime listens, and only then raise max_tokens. Accept on finish_reason rates, not a vibe that the model is “a bit wordy.”

## GEN-007. Mobile wants the conclusion in two sentences, but the model often runs to a few hundred tokens. Which layer do max_tokens, stop, templates, and length penalties each pinch, and what does over-squeezing break?

**Answer:** Control length in layers. The inference layer uses max_tokens as a hard cap and stop sequences at a paragraph end or a closing field brace. The prompt layer uses “conclusion first, exactly two sentences” and fill-in templates to cut free-form drift. The decode layer can apply a length penalty or down-score over-long completions. The product layer can return a short answer and expand on demand. “Be brief” is not enough: chat alignment prefers thoroughness. Squeeze too hard and you drop hedges and citations, so factual error rises. Critical tasks should keep the shortest necessary subordinate clause, and regressions should track lost information, not only a falling mean token count.

**Deep dive:** max_tokens will cut mid-sentence; the client needs an “incomplete” state or a continuation, and a half sentence must not be treated as the final conclusion. stop works well for lists and JSON; for prose you need a sequence that will not false-trigger on abbreviations. Templates turn open completion into slot filling (lowest variance) but a slot can still be a long paragraph. Length penalties change the decode objective, not the prior that thorough equals helpful; extreme penalties yield repeated short clauses or early EOS. A two-stage “write long, then summarize” keeps information and costs an extra forward pass, which hurts mobile time-to-first-conclusion. Token count ≠ character count: Chinese, English, digits, and markdown have different densities, so product SLOs should use characters or sentences. Score compression ratio together with retention of key hedges so you do not ship “short and backwards.”

**Follow-ups:** “Two sentences now, expand on tap” versus “generate long and truncate” — how do cache and billing differ? When stop and EOS both exist, who wins, and how does the learned end-of-turn fight a hard cut? Does a per-field character budget live in the schema or in the decoder?

**Extended questions:**
- On one question set, add max_tokens=64, a two-sentence template, or a length penalty: which is most likely to drop a negation?
- How do you detect the evasion of looking short while packing dense jargon?
- Under streaming, how do you flush the conclusion at the first sentence end without letting later tokens overturn it?

**Reasoning:** The goal is an actionable two-sentence conclusion; the constraints are a verbose alignment prior and hard cuts that break sentences. Split control across protocol / prompt / decode / product so each layer guarantees only what it can (cap, shape, preference, interaction). Then bound compression by “are the hedges still there” so the metric cannot become empty brevity.

## GEN-008. An order API needs strict JSON, but the model appends a sentence of prose after the closing brace and the parser dies. Why is “do not explain” in the prompt not enough, and can decoding lock the format?

**Answer:** Prompts are soft constraints: once sampling leaves the schema, later tokens keep improvising on a bad prefix. Low temperature and few-shot only reduce trailing prose, trailing commas, and markdown fences — they do not eliminate them. Push the legal alphabet into decoding: JSON Schema / function-call protocols, or a grammar/automaton that masks illegal tokens so natural language cannot follow `}`. The server then schema-validates and retries a bounded number of times; failures must not reach downstream. Eval should score “extra prefix/suffix” as its own bucket; field-level semantic match will miss this incident.

**Deep dive:** Constrained decoding, at each step, uses a stack or regex automaton to compute the allowed token set, sets other logits to −∞, then softmax. You pay a state transition per step and gain a syntactic guarantee. Function calling lets the runtime receive a structured argument object so the model never goes through “free text then parse.” temperature=0 can still fail on spaces, full/half-width digits, field order, or extra nulls, so decoding locks syntax and validation locks types and required fields. A repair path can slice the first complete object and parse it, but slicing can drop a warning the model meant to say — put warnings in their own field. Retries add latency; cap them and degrade to a rule template or a human. “Please output JSON” is not a sole control plane.

**Follow-ups:** A grammar can guarantee valid JSON — can it guarantee an amount field is in a business enum? On validation failure, retry the same prompt or switch to a tool call? How do you trade constrained decoding against post-hoc regex extract on first-token latency and fail rate?

**Extended questions:**
- Given an order schema with an optional nested array, which errors will a prompt almost never stop and an automaton must?
- The model emits legal JSON with a SKU that does not exist. Is that still a format bug, and which layer should block it?
- How do you build eval slices for “adds explanations / wraps markdown / trailing commas” and score them separately?

**Reasoning:** The goal is parse success as a hard guarantee; the constraint is that sampling itself drifts. Admit that prompts only change probabilities, move format into “allowed tokens per step” plus post-parse schema, and split syntax fails from semantic fails in eval. If the business cannot tolerate occasional parse errors, do not stop at the prompt layer.

## GEN-009. In a ReAct loop the model invents an Observation after Action. Should you rely on the end token or register a string stop? What must streaming guard when the stop spans tokens?

**Answer:** Register a stop sequence such as `"Observation:"`. An end token means “this whole turn is done”; the model may never emit it at a tool handoff and will instead fabricate the observation. A stop sequence is a caller-defined string matched on decoded text. It need not be one token and need not have been trained. On match, halt, run the tool, and write the real observation back into context.

**Deep dive:** An end token is a special vocab id—a training signal. A stop sequence is a UTF-8 string that may span tokens and is usually matched after detokenize. Whether the matched span is echoed back depends on the runtime; pin it with tests. Streaming hazard: a prefix of the stop (`Observ`) can flush first, so buffer an unconfirmed suffix until it is not a stop prefix. An overly common stop (a lone newline) cuts ordinary prose. String stops do not replace a correct template and end token; natural turn-ending still uses the end token. Tool handoff and isolating the next few-shot example are the classic string-stop jobs.

**Follow-ups:** How to match across tokens incrementally per step without rescanning the whole string; how streaming guarantees a partial stop never reaches the user; whether the returned text should strip the matched sequence.

**Extended questions:**
- Draw the ReAct state machine and explain why the tool boundary is a string stop rather than an end token.
- Which legitimate outputs die if the stop is a single newline?
- List paired streaming / non-streaming cases that must yield identical final text.

**Reasoning:** Separate the two stops by who defines them, which layer they fire on, and whether the model had to learn them. A tool boundary is a protocol, not a natural end of thought, so the caller supplies the string. Add a stream buffer so a half protocol marker never leaks. Accept when traces halt after Action.

## GEN-010. A nominal 8K window already holds a 2K system prompt, 4K history, and 3K retrieval, and you still need a reply. Why is the budget already blown, and how do you recut it?

**Answer:** 8K is the cap on system prompt + history + retrieval + the current user turn + reserved output, not “inputs may use a full 8K alone.” 2+4+3 is already 9K before the live question and the answer, so the server truncates or rejects. Even if it fits, middle spans are easier to ignore. Lock an output reserve first (e.g. 512–1024 tokens), slim the system prompt, slide or summarize history, and cut retrieval by score into whatever remains.

**Deep dive:** Tokenizers spend more tokens per character on Chinese, code, and tables than on English prose, so 4K characters can far exceed 4K tokens—budget in tokens, not characters. Anything clipped never happened. Prefill cost grows with input length; attention can go superlinear on long sequences. Stuffing retrieval from 2K to the ceiling adds noise and lost-in-the-middle errors, so “fill the window” is not the objective. Watch truncation rate, input-token percentiles, and accuracy when evidence sits at the front, middle, or end.

**Follow-ups:** Whether the output reserve counts inside the window; whether the API subtracts max_tokens from the window; how a length sweep finds the effective window for this task rather than the advertised one.

**Extended questions:**
- For 32K contract QA, write token quotas and a truncation policy for system prompt, clauses, question, and output.
- Which constraints does a running history summary drop, and how would structured memory put them back?
- When retrieval returns many hits, do you grow the window or rerank and send less—what metric decides?

**Reasoning:** The constraint is window = every visible token + output reserve. Protect the answer and the system rules first, then spend leftover tokens where they move accuracy. Accept with truncation logs and position-sensitive eval, not the sticker 8K.

## GEN-011. Full attention blows up training memory at 32K tokens on a 200-page contract. Should you force a long window, switch to sliding or sparse attention, or retrieve then generate?

**Answer:** Dense self-attention is O(n²) in sequence length; materializing attention scores at 32K reaches GiB-per-head scale and becomes impossible across many heads and layers. Blocked attention at inference cuts HBM traffic and peak intermediates; the arithmetic is still roughly quadratic. If the job is to pull a few clauses from known sections, retrieve and stuff a short window—cheaper and easier to evaluate in pieces. Use a long window or sliding/sparse attention only when you truly need global comparison or cross-chapter dependence, and judge it on your own long-document eval, not on whether the API accepts the full file.

**Deep dive:** A full FP16 [n,n] score matrix is about 128 MiB per head at n=8192 and about 2 GiB at n=32768, so training must chunk, rematerialize, or shrink the visible range. Doubling n roughly quadruples QKᵀ FLOPs. A sliding window of width W makes work O(n·W) and lets KV live in a ring of W, but a precise definition on page 1 often only reaches later tokens through stacked layers and is easy to miss. For document QA, chunk by heading or clause, keep the top evidence in a roughly 4K–8K window, and score misses separately: retrieval miss vs generation fabrication. A nominal 128K window only means positional encodings and server limits were lifted, not that a mid-document exception is still used.

**Follow-ups:** Does blocked attention make complexity linear? After 32 sliding-window layers, what is the theoretical receptive field, and why can the model still miss the opening definition? If you combine a long window with retrieval, do you book the extra cost under prefill or under recall?

**Extended questions:**
- Estimate how many times dense-attention FLOPs and KV cache each grow when n goes from 8K to 128K.
- When chapter 3’s definition and chapter 12’s exception must be read together, at which step does a pure sliding window drop the constraint?
- How would you use an evidence-position sweep to tell “it fits” from “it is used”?

**Reasoning:** Ask whether the task is locate, compare, or synthesize the whole document, then whether the bottleneck is training intermediates, decode KV, or quality. Quadratic cost sets the price of swallowing the file; when evidence is sparse, changing the task shape (retrieve) beats changing the kernel. Accept the choice with length and position sweeps, not with HTTP 200.

## GEN-012. An 8k window must cross-check three risk disclosures in a ~200-page prospectus. Retrieve pages and read the source, or build a hierarchical summary first? What does each approach drop?

**Answer:** Cross-checking needs clause numbers, hedges, and cross-section pointers, so chunk (by heading or page, with overlap), retrieve the relevant source spans into the 8k, and align on the original text — do not summarize and then reason. Hierarchical summary / map-reduce is fine for “list risk themes” and will wipe short hedges like “except…” and “see section 11,” producing false agreement. Two hundred pages is typically 10⁵ tokens, so an 8k window cannot swallow the file in one shot. A long-context model can cut fewer times, but you still keep low-relevance pages out of the window and attach a citation to every claim so the model cannot invent page numbers on compressed sentences.

**Deep dive:** Chunks that are too large fill the window with noise; chunks that are too small split a conditional clause — overlap exists to protect those clauses. After retrieval, rerank so you keep spans that corefer the three disclosures, not a mixed top-k of hot but unrelated passages. The map stage can ask in parallel “does this span mention buybacks / litigation / related parties” and reduce into a merged list — that solves coverage, not “does section 3 contradict section 11,” which needs both source spans in one window. Sliding over the whole file can scan it, but without a retrieval pointer the model will not remember a number from 80 pages ago. Accept on “can cite the sentence plus clause id,” not on whether the summary reads smoothly. Under a tight budget, retrieve the three loci and read them; do not uniformly summarize 200 pages.

**Follow-ups:** Should chunk size follow the embedding model’s window or the generator’s? If two loci phrase the same risk differently, how does retrieval avoid a miss? If you must summarize, how do you force numbers, negations, and cross-references to survive the summary?

**Extended questions:**
- Design a labeled set for “are three mutual citations consistent,” and explain why a topic-summary score would look better than it is.
- A long-context model ingesting all 200 pages can still drop a needle in the middle pages — do you fix that with retrieval or with reranking?
- When citation links are missing, how do you stop the model from emitting plausible but nonexistent clause ids?

**Reasoning:** The task constraint is aligning original sentences across sections, not producing a readable digest. 8k versus 10⁵ tokens forces a choice of what enters the window; compare source spans versus summaries on whether hedges and pointers survive, and pick retrieve-then-read. If acceptance becomes “write a one-page investment brief,” hierarchical summary becomes the main path. Validate with citation hit rate and contradiction recall, not summary BLEU.

## GEN-013. Pretraining packed sequences at 4K, then serving interpolates positions and advertises 64K. Short continuations still read fine, but evidence parked at token 40,000 is missed — besides compute and KV, which three facts about the training distribution stop a 4K model from simply working at 64K?

**Answer:** The model only optimized next-token loss at positions 0…4095, and most documents packed into that 4K window are much shorter than 4K, so the optimizer almost never saw supervision that required looking more than ten thousand tokens back. Position interpolation only makes index 40000 well-defined in the formula — an absolute table does not go out of range, rotary phases are scaled back into the trained angle range — it does not create long-range alignment signal. Truly long, internally coherent documents (whole books, whole repos, multi-chapter contracts) are scarce relative to short web snippets; packing many short docs into one 4K with a document-block-diagonal mask teaches “far away is another document, ignore it,” the opposite of “the evidence is at 40K.” So 64K inference is out of range in length, in position, and in dependency distance at once; a kernel that can multiply 64K keys is not a model that can use them.

**Deep dive:** Keep three facts separate. (1) Length histogram: with fixed-window packing, a 500-token page occupies a slot and a 20-page PDF is often truncated; hidden states at t≥4096 never entered the LM head, so after interpolation those positions are a new manifold. (2) Position geometry: a learned absolute vector has no row 40000; rotary positions scaled by L_train/L_target compress nearby resolution and can still leave distant phases off the trained distribution. Either way you changed “can the formula produce a score,” not “does the score still mean the relative distance seen in training.” (3) Data and packing: long coherent documents are a thin slice of tokens; concatenating to fill 4K and masking across document boundaries teaches the model to treat distance as noise. Lost-in-the-middle is close to the default on that distribution — middle-of-window positions are rarely trained as “keys you must retrieve” or as “queries that must carry distant evidence.” Raising L to 64K without upsampling long documents or adding needle/cross-section objectives leaves empty capacity. This is not the same question as prefill going quadratic or decode reading a larger KV: even if compute and memory are plenty, 4K supervision cannot support 64K retrieval. The fix is mid-training continuation (upsample long docs, allow cross-section visibility, use needle/multi-hop targets), not only raising max length on the inference API.

**Follow-ups:** If you switch to a linear-bias position scheme but still pack 4K short documents, does the needle at 40K heal by itself; how you would change packing so the model cannot learn “other documents are irrelevant”; what fraction of tokens needs to come from genuine documents longer than 16K before needle tasks start to move.

**Extended questions:**
- Draw a length histogram where 90% of documents are shorter than 1K and then packed to 4K; mark which dependency distances actually appear in the loss.
- With position interpolation only and no extra long-doc data, which probe should still pass (local copy) and which should fail (ID retrieval at 40K)?
- Why is retrieve-then-stuff 2K of evidence often more stable on a 4K checkpoint than opening a 64K window?

**Reasoning:** Write the training support as {length ≤ 4K} × {positions seen} × {dependency distances seen}. 64K inference leaves that support on all three axes; interpolation repairs only the position formula. What is missing is long-document sampling and an objective that rewards distant retrieval, not a larger attention kernel. Validate with the pair “local continuation still fluent, far needle fails,” not with whether the API still throws a length error.

## GEN-014. After you turn KV cache off, why does emitting the 2048th token suddenly become extremely slow?

**Answer:** Each autoregressive step dots the current Q with every past K and then weights V. KV cache keeps those already computed K and V in memory, so a new step only projects Q/K/V for one token and reads the cache. With the cache off, emitting one token reruns QKV projections and FFNs over a prefix of length t; at step 2048 the linear-layer work is about 2048× step 1. Attention is still O(t) per step even with a cache—that cost is one read of historical K and V. Without a cache, projections and FFNs also repeat as O(t), and that is the order-of-magnitude regression.

**Deep dive:** With cache, a decode step has new Q/K/V of shape [B, 1, d], cache [L, 2, B, n_kv, t, d_h], and the FFN touches that one position. Without cache the input is [B, t+1, d] and every layer GEMM is over sequence length t+1. Generating n tokens: projections/FFNs total ~O(n) with cache and ~O(n²) without. Illustration: t = 2048, d = 4096, FFN inner width ~1.1×10⁴; one FFN without cache does ~2048× the matmul of a single token, and 32 layers make one decode step look like a short prefill. Edge cases: a long prompt already makes prefill expensive—cache saves later decode, not prefill. Cache does not make attention linear. If the implementation cached only the last layer or only K, behavior sits in between: typical logs show normal time-to-first-token and time-per-output-token that grows linearly with length.

**Follow-ups:** Does prefill write the cache and decode read it, or both? Does the cache store projected K/V or post-softmax attention weights? In multi-turn chat, how does prefix reuse avoid prefill-ing the history again?

**Extended questions:**
- Sketch tensor shapes at step t with and without cache.
- Why is it correct to say “attention still runs with a cache; you just do not recompute historical K/V”?
- If you quantize the cache to INT8, how do latency and memory change, and is quality risk more on K or V?

**Reasoning:** The goal is that every step must see the full prefix, so the state you must keep is K and V. Compare storing them with recomputing them: recompute turns projections/FFNs into a sum over length, so step 2048 explodes. Plot step time versus output length; a near-linear slope means the cache is not in effect.

## GEN-015. You cannot retrain and KV memory is blown. Do you quantize first, drop tokens first, or offload to CPU first?

**Answer:** With frozen weights and a frozen layout, start with the KV quantization that hurts visible history least (BF16 → INT8 or FP8, watched on long-context tasks), then offload cold blocks to CPU, and only then drop old tokens to shrink the window. Quantization cuts bytes per element, offload cuts GPU residency, eviction changes what the model actually sees—these are not equally safe. Needle-in-a-haystack and multi-hop references are most sensitive to eviction and usually more tolerant of INT8 KV. Infinite streaming sessions that must keep recent detail and not OOM almost always need eviction or a sliding window.

**Deep dive:** What you can do at inference: (1) Quantize K/V per head or per token. K often has a wider dynamic range than V, so naive symmetric INT8 can drift attention scores; people scale K per channel or quantize V first. (2) Paged allocation fixes fragmentation, not theoretical size. (3) Move cold blocks to CPU/NVMe; a cold hit becomes PCIe-scale latency (tens of microseconds to milliseconds). Throughput drops but the visible token set does not. (4) Keep a few attention-sink tokens plus the last W, or evict “unimportant” tokens by historical attention scores—if a later sentence refers to an evicted entity, facts break. Train-time grouped attention, low-rank latent KV, and cross-layer sharing are out of scope here. Gate on memory, tokens/s, and long-context accuracy together; memory alone selects configs that run and cannot answer.

**Follow-ups:** Why is K usually harder to quantize than V? Are paged allocation and compression the same thing? After CPU offload, what access pattern produces p95 latency spikes?

**Extended questions:**
- Design an experiment that checks whether score-based eviction drops constraints that appeared at the start of a document.
- INT4 KV and dropping about half the history both save roughly half the memory. Which do you pick for contract QA?
- How much memory does prefix reuse of the system prompt save, and does that count as compression?

**Reasoning:** Constraints: frozen weights, change the visible token set as little as possible, GPU memory must fall. Sort methods by whether they rewrite history: quantize/offload keep the set, eviction does not. Then order pilots by implementation cost and failure mode, and gate on long-context evals rather than short-chat scores.

## GEN-016. A 13B was trained only at 4k; this week ~12k contract chunks must enter one window with no retraining budget. Position interpolation already hurt short context. At inference you fold far relative positions into groups and keep fine-grained ids nearby. How do you accept exact clause lookup, and when do you still fall back to retrieve-then-read?

**Answer:** Leave weights alone and change only the **relative indices** inside attention: a near window (illustrative W=1k–2k) still uses trained fine distances; beyond the window, divide distance by group size g (illustrative g=2–4) and clip into the 4k range so a 12k span reuses trained phases. The model can “run” 12k; that does not mean a clause at 11k still aligns with a definition at the start—positions collide. Needle exact-match plus a short-context regression first; if the needle fails, disable folding and send only 2k–4k of retrieved evidence.

**Deep dive:** Trained rotary or relative positions cover a range like 0…4095 (your max train length). Opening the window to 12k is extrapolation: phase leaves the trained manifold, and even short continuations can break—that is why interpolation already rolled back. Grouped folding does not retune RoPE base and does not train: the “j relative to i” index used in scores becomes floor(|i−j|/g) when |i−j|>W (plus true distance inside the window; the implementation must not let g smear neighbor resolution). At 12k and g=3 the farthest folded index lands around 4k and the kernel still runs. The cost is far resolution: two clauses 3g apart may share nearly the same folded index, so the model cannot tell “article 3” from “article 40” by position, only by lexical content. Split eval: (1) original tasks shorter than 4k must not sit below the pre-interpolation rollback point; (2) place the same gold clause at ~2k (inside the window), ~6k, and ~11k, wording fixed, report exact hit; (3) one item that must see a leading definition and a trailing exception together. Fail (1) and you do not ship; pass 2k and fail 11k means folding only bought fake capacity. Then the right fallback is retrieval: chunk, recall, put 2k–4k of evidence in a real 4k window—not a larger g, which smears far positions further. Do not compare with a model actually trained for a 32k advertised window; these weights are still a 4k model. Do not replace clause exact match with perplexity.

**Follow-ups:** Which of W and g first hurts “copy a nearby table,” which first hurts “the exception at the end”; KV is still allocated at true 12k length—why folding does not save HBM; as decode grows past 12k, do you remap K already written.

**Extended questions:**
- Plot true relative index versus folded index for |i−j| from 0 to 12k and mark the 4k train cap.
- How does a three-placement experiment that moves only the gold span prove a miss is position collision rather than a retrieval miss?
- What should g=1 (no folding) and g=8 do on the short-context regression, and how is that written as a control gate?

**Reasoning:** Read 12k without retraining; position codes are only defined inside 4k and far folding collides. Keep resolution in a near window, fold far distances back into range, and let needle exact match decide. If that gate fails, return to retrieve-then-read rather than treating folding as a substitute for long-context training.

## GEN-017. Nine sessions together still fit the KV budget, but each reserved a contiguous 32K buffer, so the tenth allocate fails. After you switch to a fixed block table, which waste do shared prefixes, intra-block holes, and preempt-and-swap each remove — and when do they spike p95?

**Answer:** Contiguous reservation carves max_seq_len for every request, so nine chats that are only about 8K still occupy 32K slots and the tenth fails while the ledger still shows free bytes. A block table of B tokens per block (16, for example) maps only ceil(len/B) blocks and removes the external fragmentation in the reserved tail. A system prefix lets many requests’ tables point at the same physical blocks; the first writer that diverges copies that block (copy-on-write), which removes duplicated prefixes. When the pool is tight, a policy pauses one sequence: swap its blocks to CPU, or drop them and re-prefill — that stops live requests from pinning the pool forever. The costs are a partial last block per sequence, bursty COW copies, and a whole-batch wait when some decode step hits a swapped block; p95 stops looking like “slow math” and starts looking like “waiting on PCIe or recompute.”

**Deep dive:** The block table maps logical position i to physical block `table[i // B]` and offset `i % B`. The attention kernel gathers through the table; the full KV span need not be contiguous. Small B means more indirection and more table entries; large B means a bigger hole in the last block: B=16 and length 17 occupy two blocks, 15 slots empty, billed at per-token KV bytes. Prefix sharing materializes a 1K system prompt once for 64 concurrent chats; if token 1001 differs per session, only that block is allocated (or COW’d) and the shared 1K stays a read-only alias. If a tenant changes the system prompt or injects a different retrieved span, sharing breaks for that request only — never mutate another tenant’s physical blocks. Preemption is a scheduling policy, not a compression algorithm: when free blocks cannot hold the next token or a new prefill, pick the lowest-priority or most idle sequence and memcpy to pinned host memory or discard. Swap rides PCIe at tens of microseconds to milliseconds per block group; a decode batch that includes one cold sequence stalls everyone at the sync point. Do not preempt mid-turn on an interactive tool call, or the user thinks the model died. Plan from the gap between “allocated blocks × B” and “live tokens” — that is internal fragmentation. Do not explain this allocate failure with “remaining GiB ÷ full-window KV”; it failed the reservation policy, not the theoretical byte formula.

**Follow-ups:** After two requests share a 1K prefix and diverge at token 1001, how many blocks are newly allocated versus copied; if you always preempt the newest sequence, why a 200-token high-priority completion can starve; how you read internal fragmentation from allocator counters.

**Extended questions:**
- For a mix of 200-token chats and 8K retrieval prompts, who wastes more slots at B=8 versus B=128, and who pays more gather overhead?
- Write a copy-on-write rule so that appending to a shared prefix can never mutate another tenant’s K/V.
- When would you rather drop a preempted sequence’s KV and re-prefill than swap it to CPU?

**Reasoning:** The failure is the allocation policy, not the per-token byte formula. Split waste into reserved tails, duplicated prefixes, and last-block slack, and attack them with paging, aliases, and accepting internal fragmentation. Define the action under pressure as a preemption policy and account swap/recompute tail latency separately. Validate with allocated slots versus live tokens, and with p95 when swap fires — not with whether you can still start one more full-window request.

## GEN-018. Weights already fill the GPU and 24 GiB remain. How many concurrent 8K contexts fit? Estimate with the per-token KV formula.

**Answer:** Bytes per token ≈ 2 × layers × KV heads × head dim × bytes per element. Example: 40 layers, 8 KV heads, head_dim = 128, BF16 (2 bytes) → 2×40×8×128×2 = 163840 bytes ≈ 160 KiB/token; one 8K sequence is about 1.25 GiB. 24 GiB leftover is about 19 full 8K sequences on paper; after activations, fragmentation, and runtime reserve, production often caps around 12–16 concurrent sequences. Concurrency, context length, and KV precision share one budget: stretch the window and you must drop concurrency.

**Deep dive:** Use KV head count, not query head count: grouped attention with 32 query heads and 8 KV heads caches 8. Budget: GPU memory − weights − driver context − activations ≈ KV available. Decode activations are small (sequence dim 1); prefill activations are large, so plan a prefill peak, not only steady decode. Switch to 32 KV heads and each token becomes 640 KiB, an 8K sequence ~5 GiB, and 24 GiB holds about 4 sequences. Edge cases: size traffic by p95 length, not the mean, or the tail OOMs. Paged allocation cuts fragmentation, not bytes per token. With tensor parallel degree p, divide the per-card formula by how KV is sharded. The planning curve is a hyperbola: batch × ctx ≤ available memory / bytes per token.

**Follow-ups:** If KV heads go from 8 back to 32, how does the concurrency cap move? Why can you still OOM after planning on a 2K average context? How do you rewrite the per-card formula at tensor-parallel size 2?

**Extended questions:**
- After dropping BF16 KV to FP8, how many 8K sequences fit in 24 GiB, and what quality risk do you watch?
- On the same leftover, when do you pick 4 concurrent × 32K versus 16 concurrent × 8K?
- How do you use the framework’s reported KV-block occupancy to check the formula for missing terms (for example a position-encoding lookup table)?

**Reasoning:** Maximize batch × length under a fixed leftover. Write the per-token byte formula, plug in the architecture, get one sequence’s footprint, integer-divide, and keep a peak reserve. The identity batch × ctx = constant is why the two knobs fight. Check allocated KV bytes against the formula.

## GEN-019. Which formula gives the byte footprint of KV cache when the architecture is GQA?
**Answer:** For a model where every layer is full attention and the K and V dimensions match, the approximate byte count is 2×number of layers×total cached tokens×number of KV heads×head_dim×bytes per element. Use the KV head count, not the Query head count.
**Deep dive:** Also add block alignment, quantization scales, temporary buffers, and parallel-layout overhead. Sliding-window or heterogeneous layers need a per-layer calculation.
**Follow-ups:** Why can you not estimate every model from hidden size alone? In GQA/MQA the KV head count may be smaller than the Query head count, and heterogeneous layers and parallel layouts also change actual storage.
**Extended questions:**
- With a mix of sliding-window and full-attention layers, how should you estimate the KV upper bound layer by layer?
- How do KV sharding or replication under tensor parallelism change a single-GPU capacity estimate?
**Reasoning:** First pin down the attention structure and the parallel replication pattern, then compute available concurrency.

## GEN-020. What does nominal context length measure versus effective context length, and why must the latter be measured on the task rather than copied from the spec?
**Answer:** Nominal length is the range the interface or architecture is allowed to accept. Effective length is the information range that remains reliably usable under a given task, quality threshold, and distractor conditions. The latter must be measured.
**Deep dive:** Plot quality curves by length, evidence position, and task complexity; do not only record whether a request can be accepted.
**Follow-ups:** Does a 128K input that does not error prove reliable 128K reasoning? No. That only verifies interface capacity. You still need to test evidence use and task quality at that length.
**Extended questions:**
- How would you define effective context length separately for retrieval, aggregation, and cross-document reasoning on the same model?
- When quality drops as length grows, how do you rule out an insufficient output budget or silent input truncation?
**Reasoning:** Capacity is necessary; task accuracy is what shows that the information was actually used.

## GEN-021. Locating one planted fact in a single-needle retrieval setup can still leave long-context competence unproven. Why is that test not enough?
**Answer:** A single-needle test mainly checks whether the model can locate explicit information. Real tasks also include combining multiple pieces of evidence, resolving distractors, aggregation, and multi-hop reasoning. Add multi-needle cases, repeated entities, negated information, and no-answer samples.
**Deep dive:** Fix the generation budget and record input length so differing budgets cannot hide degradation.
**Follow-ups:** If retrieval is correct but the summary is wrong, which stage should you blame? First check that the evidence was passed in complete form, then localize the aggregation or reasoning stage. Do not automatically attribute it to retrieval failure.
**Extended questions:**
- How do you construct multiple related needles, similar distractor needles, and no-answer cases so the test does not only reward string matching?
- If performance collapses after entity names are swapped in the same long-context test, what shortcut might that expose?
**Reasoning:** Score finding evidence and using evidence separately, instead of letting a single pass rate stand in for overall ability.

## GEN-022. Why can long-context quality still fall short after you change only the RoPE scaling knobs?
**Answer:** Position scaling changes the range over which positional representations are used, but the model may still lack long-sequence training and task adaptation. Attention computation, KV capacity, and serving limits also matter.
**Deep dive:** Test both long-input ability and the original short-context ability; do not only check that the configuration can be loaded.
**Follow-ups:** What do interpolation and direct extrapolation each change? Interpolation maps positions from a longer sequence into a more familiar range; direct extrapolation uses positions outside the training range. Both still need task-level validation.
**Extended questions:**
- After the window is extended, if short-context quality drops, how should you design a controlled experiment to locate the cause?
- How do you distinguish a model's failure to generalize over positions from a server-side length limit and from insufficient KV memory?
**Reasoning:** A configuration that allows a longer length is not the same as weights that have learned to use information in that range.

## GEN-023. How do you set up a scan over evidence placement so you can tell if the model drops facts that move through the context?
**Answer:** Keep the question, answer evidence, and distractor content essentially unchanged. Move the evidence to the beginning, middle, end, and several random positions, then compare accuracy and citation accuracy.
**Deep dive:** When changing position, control total length and local formatting, and sample multiple times to report uncertainty.
**Follow-ups:** What confounders arise if you copy the evidence to both the beginning and the end? That also changes how many times the evidence appears and the total length, so you cannot attribute the performance change to position alone.
**Extended questions:**
- When moving evidence, how do you preserve headings, paragraph transitions, and local formatting so you do not introduce new prompt cues?
- How can multi-position, multi-random-seed trials separate a stable position effect from sampling noise?
**Reasoning:** Only after content and length are controlled can quality differences be more confidently attributed to position.

## GEN-024. What source back-check and targeted probes show a summary of a long document still contains the exceptions that matter?
**Answer:** Build a coverage checklist of clauses, exceptions, times, and entities, then check the summary against the source item by item. Separately test negation sentences, limiting conditions, and cross-paragraph dependencies.
**Deep dive:** A fluent summary, or one similar to a reference summary, does not guarantee that every critical constraint was retained.
**Follow-ups:** How do you stress-test the case where the exception sits in the middle of the document? Hold the clauses and the question fixed, move the exception to different positions, add distractors, and check whether the summary still keeps the conditions under which the exception applies.
**Extended questions:**
- How do you design a summary counterexample where the main rule is correct but the exception is omitted, to test whether the scorer is sensitive enough?
- After cross-paragraph qualifying conditions are summarized separately, how do you verify that their logical relationship is still preserved?
**Reasoning:** Treat the key facts needed for downstream decisions as the acceptance criteria, rather than judging by writing quality alone.

## GEN-025. On turn 12 the bot asks again for an order ID the user already gave, and the window is not full. How do you separate truncation, summary drift, and retrieval miss, and what should long-term memory look like?

**Answer:** If the window still has room and a stated fact is gone, inspect the application: a rolling summary overwriting the source, a corrupted slot, or retrieval that never recalled the early turn—before blaming the model’s memory. Store order IDs, promises, and open actions as structured fields with timestamps and source turns; each turn inject only the relevant slots plus the last few raw turns. Do not keep crushing the whole transcript into a shorter prose summary. If you already summarize, diff the turn where the ID first appeared against the current summary to see whether compression dropped the field or the prompt never put memory back in.

**Deep dive:** Three pipes need separate checks. Turn- or token-truncation deletes early turns outright, but leftover window budget usually rules that out until you dump the prompt that actually reached the model. Summaries often drop numbers and negations (“out of warranty” becomes “under warranty”). Embedding memory keyed only on the latest utterance will miss an ID from turn 2 because it does not look like the current question. Structured memory updates by merge: unmentioned fields stay. Identifiers should use keyword or exact match, not vectors alone. Evaluate with probes of the form “on turn N, restate the entity from turn k,” not single-turn fluency.

**Follow-ups:** After the same dialogue is summarized three times, how do numbers and negations typically degrade? When the user corrects an order ID, how do you retire the old value? If a prompt-injection writes into memory, how do you authenticate or filter writes?

**Extended questions:**
- Design a dialogue memory table (fields, update rules, injection template) so the order ID has exactly one current value at any time.
- How do probes tell “the model saw it and ignored it” from “it was never in the prompt”?
- In a sensitive conversation, which fields must never enter long-term memory?

**Reasoning:** Get the real prompt and ask whether the fact is in the model’s visible range. If not, fix truncation, summary, or retrieval; if yes, check whether instructions require slot reuse and whether unrelated turns diluted attention. Structured merge beats prose summaries because fields can be updated and expired. Close the loop with entity-restatement probes and prompt logs.

## GEN-026. Offline RAG accuracy is 82%, but about a quarter of “correct” answers have no support in the retrieved passages. How do you evaluate unanswerable items separately and put reject gates on both retrieval and generation?

**Answer:** That accuracy is inflated by lucky guesses and parametric fill-in. Hold out a set where retrieval is deliberately missing the answer; the correct behavior is to refuse or say the sources are insufficient. On retrieval, gate on relevance and coverage: if the top score is low or key entities do not match, do not generate. On generation, require citations and an entailment check on numbers and entities; refuse if the citation cannot carry the claim. Ship with answerable accuracy, unanswerable-refusal, and false-refusal—not one blended accuracy.

**Deep dive:** Build unanswerable items by stripping evidence, swapping in same-topic distractors, or asking facts outside the document’s time range. Training and prompts need matching refuse examples, or next-token loss pushes the model to emit some span anyway. Do not retrieve on cosine alone; reconcile proper names and numbers with keywords. After generation, align spans: every number, date, and entity in the answer must be locatable in a citation. The two gates do different jobs: retrieval stops empty context from being invented; generation catches fabrication given context. Online, watch refusal rate and user follow-ups: too little refusal often means guessing, too much means the gate is tight.

**Follow-ups:** The user asks for a price and you only retrieved a product description—answerable or not? Entailment models err too; how do you use them as an aid rather than the only judge? If you mix unanswerable items into training, do answerable answers become too short or too refusal-prone?

**Extended questions:**
- Write annotation rules for several dozen answerable and unanswerable items, including how to label partial evidence.
- How does refusal rate differ between a “say you don’t know” prompt and the same prompt plus citation checks?
- How do you stop raters from marking a plausible-sounding answer correct and thereby hiding unsupported answers?

**Reasoning:** In production a wrong answer costs more than a refusal, so unanswerable items are first-class. The audit finding “marked correct, no support” implies gates on both retrieval coverage and generation entailment. Release on the triple: answerable-correct, unanswerable-refuse, false-refuse.

## GEN-027. An earnings summary’s ROUGE looks fine, but it contains a revenue figure that is not in the source. How do you catch this by aligning entities and numbers back to the document, instead of adding “do not fabricate”?

**Answer:** ROUGE only rewards n-gram overlap; a fabricated number of the same magnitude can still score well. Extract numbers, proper names, and up/down direction from the summary and force a locate-in-source check; failed locate or flipped polarity is a factual error and the sentence is rewritten or dropped. Decode with low randomness, extract evidence sentences before compressing, and keep high-risk figures in source wording with a pointer. “Do not fabricate” in the prompt only shifts the prior; it is not acceptance.

**Deep dive:** Typical shapes: a number attached to the wrong line item, a forecast written as actual, “excludes” written as “includes.” A workable pipeline is section-wise summary, merge, then a fact check. The checker need not be a large model: regex out numbers and fuzzy-match them; dictionary-check rise/fall polarity; a number that first appears in the summary and never in the source must not auto-pass. Low temperature or greedy decoding reduces drift from paraphrasing numbers but does not stop the model from inserting “common sense.” Report factual consistency: atomic claims in the summary that the source entails. Add footnote and exception items, because fluent summaries most often drop negations and limits.

**Follow-ups:** The source has two conflicting revenue definitions and the summary keeps one—is that hallucination? Does extractive summarization remove fabrication, and at what cost? If the checker misses “source is juxtaposition, summary is causation,” what extra check do you add?

**Extended questions:**
- Write a minimal number-alignment script that extracts every number from the summary and reports those that do not hit the source.
- How do you build a case where the main claim is right and a footnote exception is dropped?
- Which summaries should human review prioritize, instead of a fixed random percent?

**Reasoning:** Readers will act on the numbers, so numbers and entities must be sourceable. ROUGE does not measure that, so you add an alignment check. Auto-block what you can align; send the rest to humans. Prompts change the prior; they do not replace the check.

## GEN-028. A user stuck on “enter the return tracking number” suddenly asks if the black SKU is in stock. How do you detect the topic switch so the old slots stay parked and do not pollute the inventory lookup?

**Answer:** Separate task state from the current utterance: classify the turn as continue / insert a new task / cancel, and do not default every sentence into the old form. On a new intent, snapshot and suspend the return—including filled slots—and answer inventory from a clean context; resume only when the user says to continue the return. Never silently pass the return ID into the inventory tool.

**Deep dive:** The usual failure is concatenating the whole history so the model stuffs “tracking number” and “the black one” into one function call, or continuing the slot from the most recent entity. Detection can stack: a rule when the expected slot type does not match entities in this sentence; a continue/switch/cancel classifier; a structured dialogue act before any tool call. The suspend stack must store task id, filled slots, and the next question—not only a prose recap. A hard case is “which tracking number do I use for this black item?”—it looks like a switch and is a clarification—so high-risk switches should ask a short confirm. The eval set must cover mid-task insert, clarification, and cancel-then-restart, scored on task completion and wrong-tool rate.

**Follow-ups:** A confirm costs an extra turn; when is silent suspend acceptable? If the user wants return and inventory in parallel, how do you isolate the two contexts? If the classifier labels a switch as continue, how does tool-argument validation catch it?

**Extended questions:**
- Write labeling guidance for continue/switch/cancel and give one easily confused example for each.
- If a suspended task is resumed after tens of minutes, do you re-validate order state?
- How do you mark topic boundaries in logs so multi-turn robustness can be measured offline?

**Reasoning:** Task slots have side effects—a bad fill hits the return API—so you must decide whether this sentence is for the current slot before you fill or suspend. Detection can be cheap; resume must be explicit. Accept on wrong-tool-argument rate and task completion, not on whether a single turn sounds fluent.
