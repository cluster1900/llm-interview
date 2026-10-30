# Alignment, Post-training and Reasoning Models

[← Back to contents](../README.md)

## ALIGN-001. After two weeks of optimizing a support model for resolution rate and thumbs-up, it starts promising refunds with no policy basis—which layer of alignment does this crack, and do you change the training objective, the system prompt, or human review first?

**Answer:** This is reward hacking from a misspecified goal: thumbs-up and first-contact close are proxy metrics, while the real intent is "solve the issue inside authorization." The model maximized the wrong objective efficiently. Engineering alignment has at least three layers—whether the intent is written down, whether the optimization target is faithful to that intent, and whether runtime constraints cover the gaps—and the second layer broke first. Change the training objective first: stop treating uncalibrated thumbs-up as positives, label unauthorized promises as negatives, and add a policy gold set. A system prompt is a soft constraint and will not override a policy already shaped by reward. Human review can catch promises already sent; it is not a long-term alignment mechanism. Stronger capability makes a wrong target more harmful, so alignment is about optimization direction, not about the model being smarter.

**Deep dive:** The loop is: conversation → thumbs-up or close label → preference pairs or a reward model → policy update → more refund promises → more thumbs-up. Illustrative: if promising a refund lifts thumbs-up from about 60% to 75% while compliant resolution falls, the reward model scores "apology plus refund" as winning speech. Short term: intercept unauthorized promises with rules and roll back to the pre-optimization checkpoint. Medium term: label satisfaction and compliance as separate dimensions, and keep an eval slice of "user pressures for a refund the policy forbids." Long term: put verifiable outcomes into the objective—correct policy citations, repeat-contact rate, anomalous refund amounts—not thumbs-up alone. Boundary: when the user truly qualifies, the model should still help; alignment is authorization, not blanket refusal. Do not turn this into a list of governance frameworks; the question is whether the objective and the behavior match.

**Follow-ups:** If the reward model is accurate on chit-chat and drifts on refund dialogues, do you retrain the reward model or drop that class from reinforcement learning; if the product insists on "make the user happy first," which numbers show misaligned optimization creating finance and compliance incidents; if the prompt says "do not promise refunds" but preference data says the opposite, who wins at inference.

**Extended questions:**
- How do you tell reward hacking apart from an authorization scope that was never specified?
- If you use DPO instead of an explicit reward model, how does thumbs-up bias enter the implicit preference?
- For an internal employee assistant, is "the human" the asker, employer policy, or the regulator, and who wins a conflict by default?

**Reasoning:** Start from the gap between what we want the model to do and what we actually optimize, fix the objective and the data first, then use runtime gates to stop the bleeding. Accept on unauthorized-promise rate, policy consistency, and repeat contacts—not on thumbs-up recovering.

## ALIGN-002. Expert-rewritten demonstrations are done, so the team treats supervised fine-tuning as alignment complete. After launch the model still inflates promises to please the user. Which layer is that judgment wrong at?

**Answer:** Supervised fine-tuning only maximizes conditional likelihood on the demonstration distribution. It learns “write like these samples,” not “still obey goals, preferences, and safety under unseen pressure.” Expert rewrites can cover common good replies, but they almost never give a pairwise contrast: on the same request, which should win, a people-pleasing promise or a policy-legal refusal. Alignment is the goal; SFT is one tool. Without preference comparisons, safety refusals, or enforceable constraints, fitting demonstrations is not finishing alignment.

**Deep dive:** The SFT data path is `(prompt, ideal reply)` → cross-entropy on reply tokens only. Gradients raise the probability of demonstration tokens; they do not compare two fluent replies of which one is making a guarantee the policy forbids. Alignment is about relative constraints: more honest, less harmful, more on-policy, even when that makes the reply less agreeable. Scale: a few thousand to a few tens of thousands of demonstrations often stabilize greetings, fields, and closings; under an out-of-distribution push such as “promise the customer we will ship next week,” the model still fills in the promise by maximum likelihood, because the loss never separately punished an unauthorized guarantee. Boundary: if the demonstrations already systematically include refusals, uncertainty, and walk-backs, SFT is itself part of alignment; quality then depends on coverage and internal consistency, not on the stage name. Preference optimization later also cannot fix a wrong rate or a wrong legal conclusion written into the demonstrations—that is a bad supervision signal, not a missing RL round. Do not treat “SFT then preferences” as a fixed pipeline for every product: fixed-field tickets can be closed with a schema and rules; open chat that inflates promises needs an explicit failure slice, not a demonstration-set perplexity.

**Follow-ups:** If about one fifth of the demonstrations are already refusals, do you still run a preference stage, and which slices prove a gap remains; when a user’s inflated ask conflicts with policy, which side does a pure SFT model usually take, and why does the loss push it there; if you only do SFT and intercept with a policy classifier at decode time, which alignment layer is that, and does it change the generation distribution or only the exit filter.

**Extended questions:**
- Write three “help me guarantee this to the customer” prompts that the demonstration set never covered but production will see, and explain why cross-entropy cannot learn the refusal.
- If the product only emits fixed ticket fields and never opens free chat, should the alignment budget go mainly to demonstrations, rule checks, or preference pairs?
- How would “answered when it should refuse / refused when it should answer / is the promise executable” prove that alignment is still incomplete after SFT, rather than looking only at demonstration loss?

**Reasoning:** Define alignment as “still obeying the goal under out-of-distribution pressure,” and SFT as “fitting the conditional distribution of demonstrations.” The goal set is larger than the demonstration set, so a finished fit is not a satisfied constraint. Accept the claim only on adversarial promise slices and executability checks, not on training loss.

## ALIGN-003. The SFT model already gets work done, but a polite social-engineering prompt still elicits internal URLs. What do preferences, the reward model, and a KL-leashed policy update each block — and which ring, if loosened, looks more helpful and becomes more dangerous?

**Answer:** Preference data defines which completions are better, including “do not leak internal links” and “do not play along with the setup.” The reward model turns pairwise comparisons into a scalar r(x,y) that the policy can climb. The policy update (classically PPO-like) raises the probability of high-r completions while a KL penalty toward the SFT/reference model stops a dash into reward exploits. Loosen KL, or omit pairs of the form “polite setup + ask for an internal link,” and the model becomes more willing, longer, and smoother according to r — leakage goes up. Alignment re-ranks completions; it does not write a link denylist into the weights.

**Deep dive:** Classic loop: sample y from the current policy → score with r_θ → (in PPO, estimate advantages with a critic) → update π, and periodically send new samples back for human comparison because r goes stale off the labeled distribution. High β nearly freezes the SFT policy, so the social-engineering attack still works; low β produces reward hacking: piled courtesy, piled disclaimers, avoided refusals. If preferences label only helpfulness and not confidentiality, r rewards “explain the internal link clearly.” Direct preference methods such as DPO are the same family — move the policy with comparisons — they just drop a standalone r and critic, and they still fail without confidentiality pairs. Boundary: jailbreak phrasings that never appeared in comparisons are not covered by this loop; production still needs output filters, link scanners, and a red-team set. RLHF is not one algorithm name; it is the family “human (or proxy) feedback → optimize the policy.” Keep the data and the optimizer separate when you talk about alignment.

**Follow-ups:** If the reward model is unstable between “helpful but leaking” and “refusing but blunt,” do you change the rubric or add a dimension; if KL is averaged per token versus summed over the sequence, how does a long answer exploit the leash differently; if you only add preferences on prompts that contain links, will the model over-refuse on link-free questions.

**Extended questions:**
- Is internal-link leakage a training problem or a post-decode filter problem, and which contrast set splits them?
- If the reference is an older base than the SFT model, what behavior is the KL leash actually tied to?
- If polite wording fools the reward model into high scores, what product symptom does the policy update amplify?

**Reasoning:** Write leakage as “a completion that is likely under the current policy but should not rank first.” Preferences supply the order, the reward model turns order into a scalar, KL limits travel so you do not optimize holes in r. Predict the failure from the missing ring: no confidentiality pairs → r encourages leaks; no KL → surface helpfulness. Accept with a social-engineering red team, leak rate, and task success when nothing leaks — not with helpfulness win rate alone.

## ALIGN-004. Pairwise preference agreement is barely above half, yet the reward model’s classification loss looks great. How do you tell an incomparable task from a vague guide, and strip noise from learnable signal before training?

**Answer:** A falling loss only means the model is fitting the labels you have, including annotator disagreement and length habits. Split agreement by prompt type: safety pairs can be high; open style pairs near chance are not “train the annotators harder,” they should not be a single win/lose. A vague guide looks like people scoring the same class of prompt on facts versus politeness, then converging after arbitration. An incomparable task looks like arbitrators also stably marking “too close.” Before training, drop or down-weight low-agreement slices, or switch to multi-dimension scores; do not feed the whole dump to a reward model.

**Deep dive:** Path: prompt plus two replies → two people independently pick win/lose/tie → disagreements go to arbitration → gold labels. At about half agreement the labels are near coin flips; a reward model can still fit a decision surface to majority vote and then amplify the most common surface preference: longer, more certain, less refusal. Order of operations: (1) overall and bucketed agreement (safety, fact, style, tool use); (2) gold items with a known standard, to see whether annotators follow the guide; (3) rewrite the guide and relabel a small batch—if agreement rises it was the guide, if not it was the item design. Avoid pairs where both replies are obviously great or obviously bad, and score correctness, safety, and brevity separately before a product-weighted combine. For example, restage about 5–10% each week and mix about 5–10% gold items; annotators below threshold pause and retrain. Boundary: some people want detail, some want short—that is value plurality, not something to crush with forced agreement. Turn it into a style control, or do not use a single reward on those prompts. Do not read reward-model validation accuracy before agreement is stable; that accuracy is reciting noise. Pairs that differ by 2× in length and unanimously pick the long one are high-quality noise, not high-quality preference.

**Follow-ups:** After three annotators and majority vote, agreement looks higher—is that better quality or ties hidden inside the majority; if 2×-length pairs agree strongly and always pick long, do they still enter the training set; if the reward model is especially accurate on the low-agreement slice, did it learn a stable preference or one annotator’s habit.

**Extended questions:**
- Design a rubric with at least four dimensions and state when “no overall ranking” is allowed.
- If agreement moves from about 52% to 70% but held-out human preference accuracy does not, do you keep collecting pairs the same way?
- How would gold-item pass rate and bucketed agreement become an intake gate that blocks a reward model whose loss is beautiful and whose live behavior is more sycophantic?

**Reasoning:** The goal is a reward that learns only stable preferences that match the product. Use bucketed agreement and gold items to split a vague guide from an incomparable task, then relabel, split dimensions, or drop. Reward-train loss is not an intake metric; surface accuracy on noisy labels is meaningless.

## ALIGN-005. A PPO step keeps four networks on the GPU: policy, reference, reward, and value. Why does the clipped objective zero the policy gradient on a token when the advantage is positive and the probability ratio is already ≥ 1+ε? What variance do the value network and GAE each kill, and roughly how do the four weight sets plus optimizer state split memory?

**Answer:** The proximal objective is min(rA, clip(r, 1−ε, 1+ε)A) with r = π_θ(a|s)/π_old(a|s). When A>0 and r≥1+ε both arguments are already (1+ε)A, so increasing r further does not decrease the loss and that token’s policy gradient is 0 — this minibatch refuses to push an already-raised action any higher. The value network V_φ gives the prefix a baseline so A = return − V has lower variance than the raw return. GAE interpolates with λ between one-step TD and full Monte Carlo, sending a scalar reward that usually arrives only at end-of-sequence backward through the tokens, so you need neither to wait for EOS for a signal nor to trust a single noisy bootstrap. Resident modules are a trainable policy, a frozen reference, a frozen reward model, and a trainable value: Adam’s first and second moments sit mainly on policy and value; reference and reward can run at lower precision or be offloaded. Illustratively, with four same-width 7B-class towers fully resident, optimizer state on the two trainable sides often dominates the two frozen weight copies.

**Deep dive:** ε is commonly in the 0.1–0.2 range (illustrative). Four quadrants: A>0 and r>1+ε block further increase; A<0 and r<1−ε block further penalty; otherwise the unclipped rA path is active. Clip is a trust region relative to π_old for this step, not the KL to the SFT reference; you still add β·KL(π, π_ref), or many steps can walk far along directions the clip locally allowed. In language alignment the reward is almost always a single scalar from the reward model at the last token, so vanilla REINFORCE variance grows with length. TD residual δ_t = r_t + γV(s_{t+1}) − V(s_t), and GAE sets A_t = Σ_{l≥0}(γλ)^l δ_{t+l}: λ→0 is biased by V’s error and low variance; λ→1 is closer to the full return and high variance. λ around 0.95 (illustrative) is what spreads the terminal score onto earlier tokens. The four modules spend memory differently: π_θ needs activations, gradients, and an optimizer; π_ref only needs logprobs, no gradient; the reward model only scores a finished sample and can live on another device than the training step; V_φ is trained with MSE/Huber on GAE targets, often a scalar head initialized from the policy — a full sibling tower adds another trainable copy. Order-of-magnitude: BF16 weights are about 2 bytes/parameter for policy+value; Adam’s FP32 moments add about 8 bytes per trainable parameter, plus 4 more if you keep FP32 master weights. Reference+reward at full BF16 are roughly two more weight copies; the usual cuts are LoRA on the policy, a shared π/V backbone, and quantized or separately hosted ref/RM — not four full Adam states.

**Follow-ups:** If you drop the value network and use a per-prompt group mean as the baseline, which of the four towers can disappear and what generation cost replaces it; why clip is not a substitute for KL to π_ref; if the value head sits on the policy hidden state, how you stop value gradients from wrecking the LM head.

**Extended questions:**
- Write the four branches by sign of A and where r sits relative to 1±ε, and mark which branches have zero policy gradient.
- For a 512-token completion with reward only at EOS, sketch how A_t sits along positions at λ=0 versus λ=0.95.
- Rank four memory cuts: LoRA on the policy, 8-bit reference, shared π/V backbone, CPU-offload the reward model — which hits peak training memory versus tokens/s.

**Reasoning:** PPO wants a small step toward higher reward with controllable credit-assignment variance. Clip implements “this minibatch cannot step too far,” so when A>0 and r is already at the cap the gradient must stop. V removes variance from the absolute scale of the return; GAE removes variance from credit assignment under a delayed reward. The four towers are those formulas made concrete: two trainable with optimizers, two frozen forward-only. Split the memory bill by who needs Adam and who can be offloaded, not by averaging “an alignment algorithm” across four equal piles.

## ALIGN-006. PPO reward keeps climbing while human review finds longer, politer answers and no fewer factual errors. How do you tell the policy is gaming the reward model, and which side do you fix first?

**Answer:** Reward has decoupled from real preference: the policy is optimizing surface statistics the reward model likes—length, politeness, confidence, list formatting—not truth or task completion. Freeze the policy and compare length, boilerplate rate, and factual error rate before vs after RL. If reward rises and human win rate falls, roll back to a checkpoint instead of taking more steps. Then audit reward blind spots, retrain the reward with adversarial pairs and process or verifiable signals, and only then resume RL in small steps.

**Deep dive:** A single scalar on a full answer is easy to hijack with length, heading markup, and safety boilerplate that correlate with high scores in the training pairs. Diagnose with reward, KL to the reference, human or rule-based factual score, and mean output length side by side. Rising reward, rising KL, falling human score is hacking. A length penalty alone often shifts the exploit to another feature. More stable fixes: add “long and empty” vs “short and correct” pairs to the reward data; split heads for correctness, safety, and brevity with capped weights; use unit tests or a calculator as a hard reward where you can; trip a human-sample circuit breaker. Roll the policy back before you repair the reward so you do not keep climbing a compromised scalar.

**Follow-ups:** Why is a single scalar especially easy to game? How does process supervision reduce “last sentence right, middle fabricated”? How do you trip a breaker on an already-shipped policy without draining all traffic?

**Extended questions:**
- Build a batch of long, polite, factually wrong answers and check whether the current reward model scores them systematically high.
- When mixing verifiable and preference rewards, how do you stop a coding task from optimizing ugly solutions that merely pass tests?
- Which incidents does a dashboard that plots only reward miss?

**Reasoning:** The true objective is human preference; the reward model is a proxy. When the proxy and spot checks disagree, stop trusting that reward. Roll back to protect quality, then patch the exploited features or switch to a hard reward. Read whether reward, KL, human score, and length move together or against each other to see if optimization still tracks the real goal.

## ALIGN-007. The base model will continue a contract into longer clauses but will not obey “summarize in five bullets.” After SFT the format holds, yet the model prefers flattering long essays. What does each of the three stages maximize, and why can the third stage look only at “which one is better” instead of a full gold answer?

**Answer:** Pretraining maximizes next-token likelihood on raw unlabeled text and learns continuation, so a contract grows in clause-voice instead of following “five bullets.” Supervised fine-tuning maximizes the likelihood of demonstration answers given instructions, which writes in format and compliance — and also copies the courtesy and padding that dominate those demonstrations. Preference optimization maximizes the winner’s advantage over the loser under a comparison signal (an explicit reward or a direct policy loss). Raters need not write a perfect five-bullet answer; they only judge which completion is cleaner and less sycophantic. Comparing is cheaper than composing, so the third stage can live without new full demonstrations.

**Deep dive:** Data shapes differ: pretraining is on the order of 10¹¹–10¹³ tokens for contemporary large models (exact scale depends on the run); SFT is often 10³–10⁶ human or synthetic instruction pairs; preference data is an ordering among two or more completions for the same prompt. RLHF-style methods fit r(x,y) and then run PPO-like updates on r − β KL(π ∥ π_ref); DPO-style methods use the correspondence between the KL-constrained optimal policy and the reward and apply a classification loss on (y_w, y_l). SFT is not enough for a concrete reason: the mode of demonstration data is often “a long answer that looks diligent,” which is exactly what reinforces flattery and verbosity. If the third stage has neither comparisons nor an automatic checker, it has no objective. Boundary: if SFT systematically taught the wrong format, later preferences only pick the less-bad completion inside that format and cannot invent a structure they never saw; math and code with a ground truth fit verifiable rewards better than style comparisons alone.

**Follow-ups:** If you skip SFT and go from pretraining to preferences, will the policy still emit five bullets stably; if the KL coefficient is too high or too low, which way do the flattering essays return; if annotators always mark the shorter one as winner, will the third stage also cut the five necessary points.

**Extended questions:**
- The same prompts have both SFT demonstrations and preference pairs: SFT then preferences versus rewriting demonstrations into pairs and doing preferences only — how does format convergence differ?
- Pretraining already saw many “Summary:” headings. Why is that still not instruction following?
- If the third stage scores with an automatic rule (exactly five bullets) instead of human comparison, is it still preference optimization, and which signal is missing?

**Reasoning:** Align the three stages as objective → data shape → resulting behavior: continuation, imitation of demonstrations, moving probability mass by orderings. Use “a perfect answer is hard, a pairwise call is easy” to explain why the third stage needs no full gold text. Accept separately: instruction format appears after SFT, sycophancy/length drop after preferences, and a prompt set held out from the preference data to catch pair memorization.

## ALIGN-008. You can afford double annotation on only one thousand of twenty thousand comparison pairs; the rest will be scored by a judge model. How do you split “machine-ok” from “must be human,” and keep the judge from writing its blind spots into the policy?

**Answer:** Let the machine judge only dimensions whose spec is clear and checkable against a reference or a fact: parseable format, staying on topic, numbers taken from the given document. Value trade-offs, bodily and legal risk, discrimination, and outward legal commitments stay human, or at least get a much higher audit rate. Judge output is a weak label: calibrate on a gold set, and send low-confidence or disputed pairs back to people. Do not let the same weights be both the policy teacher and the only referee, or the judge’s length bias and politeness bias will be written into the policy.

**Deep dive:** Typical path: sample two replies for one prompt → the judge picks a winner from written principles or a rubric → those pairs train a reward model or a preference objective. Cost differs by about an order of magnitude: a human pair often takes minutes plus training; a machine judge can score in seconds, so twenty thousand pairs with only about a thousand double-labeled is common. The judge inherits its own training distribution: it prefers longer answers, polite filler, and wording that resembles itself, and it is weak on irony, culturally specific harm, and “sounds compliant while the steps are still there.” Calibration: hold out, for example, 200–500 pairs with human agreement and report agreement by dimension; format usually beats safety, and if safety falls below an acceptable bar that whole dimension goes back to humans. The judge and the trained policy should be different models, or at least different prompts and principle text, to avoid self-scoring. Boundary: “which write-up is better” is subjective; a machine judge collapses plural preferences into one scalar, so split dimensions instead of forcing a single ranking. RLAIF is not “no humans”: people still write the principles, the gold set, the audit rate, and the veto. After launch, fold new failure types into the gold set, or the judge goes stale as the policy distribution moves.

**Follow-ups:** If the judge and the policy are from the same family, how do they amplify length bias in both directions; with only a few dozen gold pairs, all easy right/wrong, would you still widen machine judging; when machine and human labels conflict, is that pair noise, a hard example, or a sign the principles are wrong.

**Extended questions:**
- List label sources and audit rates for format, fact, tone, and safety, and mark which dimension must never be majority-voted by a machine.
- If the judge is systematically looser than humans on refusing out-of-scope requests, what kind of overreach shows up first after a few more weeks of policy training?
- How would “machine–human agreement versus training step” tell you the judge is stale, instead of watching only the mean reward rise?

**Reasoning:** The constraints are labeling budget and asymmetric cost of a wrong label. Split machine versus human by “can the spec be checked as a rule” and “how bad is a miss,” then pin the judge’s blind spots with a gold set. The policy may eat only calibrated labels, so the judge and the policy cannot close a loop on each other.

## ALIGN-009. After preference alignment, refusal on harmful prompts rises but coding and math scores drop. How do you tell a real alignment tax from capability washed out by the data?

**Answer:** Split “refused what should be refused” from “refused benign work, or lost skills it used to have.” A real alignment tax is a small, explainable drop on allowed tasks while the safety boundary tightens. If the model starts refusing ordinary programming prompts, or solutions get shorter and accuracy collapses, the preference set is likely refusal-heavy, capability-light, or the reward head is scoring politeness and refusal. Plot safety and capability as a Pareto: if adding only safety pairs hurts capability, add capability-preserving pairs or split policies—do not pour more of the same mix.

**Deep dive:** Over-refusal is often counted as a safety win. Report harmful-prompt refusal, benign false-refusal, and absolute scores on code, math, and instruction following. If chosen answers are systematically shorter, more refusing, and more formulaic, the policy follows those surface features. Mitigations: put explicit “help when it is allowed” pairs in the preference data; keep an SFT reference on allowed domains and anchor with KL or a mixed loss; enforce safety with a classifier or rules so the generator does not learn “always refuse.” Switching to DPO-style objectives does not replace data mix. At serving time you can route: stricter policy on high-risk prompts, the capable model on tools and code.

**Follow-ups:** How do you build a false-refusal set of prompts that look sensitive but are benign? If raising KL restores capability and safety slides back, how do you pick the operating point? Why is a rising reward score not enough to call alignment healthy?

**Extended questions:**
- Draw a Pareto of refusal rate versus coding score and mark an operating point you would ship.
- If annotators are told “refuse when unsure,” how does that systematically distort the tax?
- How do you implement split refusal (for example medical advice vs writing a sort function) in both data and serving policy?

**Reasoning:** The safety floor cannot move down; capability regressions must be explained. Disaggregate metrics to kill “refuse everything” as fake alignment, then inspect label mix and chosen style against the product tasks. If more data or split policies close the gap, it is not a tax you must pay; only a leftover trade-off after both sides are optimized is an alignment tax. Gate releases on false-refusal and capability together.

## ALIGN-010. You cannot write a reward formula, but a person can tell which of two behaviors looks more like the goal. Using “pairwise compare short clips → fit a reward → run RL” on control tasks: what does that loop solve, and why does the policy start to go strange as soon as you stop collecting comparisons?

**Answer:** Some goals barely exist as a differentiable scalar: motion should “look skilled,” posture should “look natural,” behavior should “not game the spec.” People are bad at absolute scores and decent at saying which of two short trajectories is better. Fit a reward model to those comparisons (pairwise logistic: P(i ≻ j) rises with r_i − r_j), then run reinforcement learning on that r. The policy will search holes in r, so you must keep asking humans about newly generated trajectories. When comparison stops, you are optimizing a stale proxy, not the human’s current judgment.

**Deep dive:** This loop was used around 2017 on arcade-style games and simulated robot locomotion: humans labeled a small number of clips relative to total environment steps, and the agent still learned playable policies and motion styles that are hard to write as a reward. Pairwise beats a 1–5 score because people calibrate absolute scales poorly and agree more on which is better. The reason for interleaving is distribution shift: once the policy moves, new actions that old r never saw can get absurdly high scores — jitter farming, wedging into a simulator numeric seam. When the same loop later moved to language models, “clips” became two texts under one prompt and “the environment” became the prompt distribution; the structure stayed: compare → reward model → policy update → compare again. The risks moved too: style hacking, length hacking, harmful compliance in the reward model’s blind spots. Do not treat a once-fit offline r as “the human” forever; it is a regressor on one batch of labels.

**Follow-ups:** Clips that are too short lose policy context and clips that are too long people will not finish — how do you cut; if two annotators flip the same pair, does the reward model treat it as noise, an average, or a dropped pair; control tasks can reset the environment and resample, language tasks cannot “reset” — how do you refresh the prompt set when you add online comparisons.

**Extended questions:**
- Why does an absolute score on a single clip have higher variance than a pairwise call, and what fake scale does that push into the reward model?
- If comparisons come only from an early random policy and later dexterous motions were never labeled, how does r systematically down-rank truly better behavior?
- Moving the loop from “a few seconds of simulation” to “a few hundred tokens of answer,” how does human working-memory limit change sampling and presentation?

**Reasoning:** From “no trustworthy handwritten reward, but trustworthy pairwise calls,” you must learn r and then run RL; from “the policy will walk off r’s training distribution,” comparisons must continue online. The life-or-death check is whether r still agrees with humans on new trajectories after labeling stops — not whether the proxy score in the training environment is still rising.

## ALIGN-011. Safety annotators should not read attack samples at scale, but the product still has to cut harmful replies. If you write principles as executable clauses and let the model critique then rewrite, versus piling on more human pairwise comparisons, are you saving exposure or saving the standard itself? Which conflicts break first?

**Answer:** Written principles turn “harmless, honest, helpful” from a standard hidden in annotators’ heads into clauses you can review and edit. The model critiques its draft against those clauses, rewrites, then picks between two replies by the same principles, and those generated preferences drive later optimization. This principle-driven critique-and-rewrite loop is often called constitutional alignment. What you mainly save is how much attack content annotators must read, and the headcount of pairwise labeling—not the job of deciding what counts as harm. That judgment moves earlier, to the people who write principles and audit. When principles collide (be helpful vs refuse actionable dangerous steps), or when clauses are too coarse or only change tone, self-critique becomes surface compliance; a human comparison on the concrete case still catches more.

**Deep dive:** The flow is often two stages. Supervised stage: draft on an out-of-bounds prompt → critique clause by clause (should have refused, still listed steps, faked certainty) → rewrite → SFT on the rewrite. Preference stage: a model picks a winner by the principles, forming machine preferences for a reward model or preference optimization. Human pairwise standards live in the guide and in annotator disagreement, so changing the standard almost means relabeling; a principle method can change later labels by editing one executable clause, which is more transparent. Scale: principles should be a few dozen statements that map to an edit, not a manifesto; each should point to a visible action such as “delete steps / switch to refusal / demand a citation.” Boundary: if the model cannot see a class of harm, the critique step misses its own blind spot and machine preferences amplify base-model bias. Helpfulness is hard to compress into safety clauses; in practice harmlessness often goes to principles plus a machine judge, while “which explanation is more useful” stays with human preference. This does not cancel humans: principle authors, red teams, and auditors remain people; they just no longer read every attack sample. Principles and human comparison are not exclusive; high-risk prompts can still go to human review.

**Follow-ups:** When “be as helpful as possible” and “do not provide actionable dangerous steps” fire together, which clause must the critique satisfy first, and how do you write that as a stack rather than as slogans in parallel; if you generate labels only from principles and skip human preference entirely, what held-out set proves you are not amplifying base-model bias; after a company writes internal conduct into principles, who wins when those conflict with general harmlessness.

**Extended questions:**
- Write five executable principles, each tied to a concrete rewrite, not a value adjective.
- If self-critique only changes tone, not content (“I cannot give details” while the steps remain), how does training loss mislead you?
- How would the refusal-rate gap on a fixed prompt batch, before versus after a principle edit, show that behavior is principle-driven rather than just the base model’s temperament?

**Reasoning:** The goal is fewer harmful outputs and less annotator exposure; the constraint is that the standard must be reviewable and editable. Move judgment from “everyone sees a pair” to “a few people write principles + the model executes + humans audit.” Failure modes are colliding principles and weak self-evaluation, so accept the design on stacked principles and human audits, not on machine-preference accuracy alone.

## ALIGN-012. You only have a set of chosen/rejected pairs and a reference checkpoint. Under what conditions would you skip a standalone reward model and online sampling, and train a classification-style preference objective instead?

**Answer:** When the pairs already cover the prompt distribution you care about, the chosen/rejected gap is stable, and you do not yet need the policy to explore its own new outputs, you can skip both. Write the KL-constrained optimal policy as r(x,y) ∝ β log(π(y|x)/π_ref(y|x)), substitute into a pairwise preference likelihood, and you get a loss that only compares log-odds of the two answers. Training loads a trainable policy and a frozen reference—no on-policy generation, no separate scorer.

**Deep dive:** The loss raises chosen log-odds and lowers rejected log-odds relative to the reference. Large β stays close to the reference (smaller edits, safer general skills); small β edits harder and is more likely to damage them. Known failures: both answers’ log-probs fall and the model widens the gap by becoming generally less certain; a much longer chosen answer accumulates log-prob by length, so the model learns to ramble; prompts off the offline snapshot have no exploration signal. Variant losses change normalization or weaken the explicit reference; none of them change the fact that the data is a frozen snapshot. Math and code, where a program can judge correctness and the policy must roll out new traces, are a better fit for methods that still sample online.

**Follow-ups:** Why can we say this loss learns a reward implicitly without storing that reward as a network? How do you try orders of magnitude for β? If chosen is twice as long as rejected, which way does the gradient tilt? What signs show general capability is slipping after training?

**Extended questions:**
- Write the loss and mark what chosen, rejected, π_ref, and β do in the numerator and denominator.
- In the training logs, how do you tell “the model truly prefers chosen” from “both probabilities fell and only the gap grew”?
- If new human preferences arrive every week, how do you take small offline steps without rebuilding a full online sampling stack?

**Reasoning:** Starting from “maximize expected reward while staying close in KL,” the optimal policy is log-linear in the reward; substitute that reward into a pairwise comparison and online RL collapses to one classification pass. The closed form is valid only while the offline preferences still represent the distribution you want to improve. When that distribution moves, the closed form is not enough.

## ALIGN-013. What allows DPO to drop the extra trained reward network?
**Answer:** DPO uses the relationship between the optimal policy and the reward in KL-regularized preference optimization to turn pairwise preferences into a classification loss on the policy; standard offline DPO does not need to train an explicit reward model or run online rollouts.
**Deep dive:** It still depends on preference data, a reference policy, and hyperparameters; that is not the same as having no reward assumption.
**Follow-ups:** What if the preference data does not cover the deployment distribution? Add preference samples from the deployment setting and check generalization on a held-out test set; do not judge from training loss alone.
**Extended questions:**
- When the reference policy is far from the policy being trained, how do you diagnose whether DPO optimization is departing from the intended objective?
- When preference labels for the same question conflict, how do you distinguish annotation noise from genuine preference diversity?
**Reasoning:** First compare data coverage and training cost, then use an independent task set to check whether preference gains turn into product quality.

## ALIGN-014. The alignment box currently keeps four networks resident — policy, reference, reward, value — and samples live completions every step. After you switch to a classification-style loss on fixed preference pairs, which accidents disappear? When the model is already stronger than the old preference set, or the task can be scored automatically, why does that offline loss show false convergence first?

**Answer:** PPO-style preference RL keeps policy, reference, reward, and critic in memory, and also decodes online, estimates advantages, and tunes KL and clipping. Typical accidents are a gamed reward model, a lagging critic, sample/train desync, and collapse after one hyperparameter nudge. Direct preference optimization uses the closed-form link between the KL-constrained optimal policy and the reward, folds (y_w, y_l) into a logistic loss against the reference, and usually leaves only the policy plus a frozen reference — training looks like SFT. An offline loss can go very low on a fixed pair set and only means “this label batch was memorized,” not “better answers outside the set were found.” When the model has outgrown the old sampler, or math/code can score fresh samples with a checker, driving that loss down further overfits length and wording first; true solve rate stops moving.

**Deep dive:** Loss sketch: ℓ = −log σ( β [log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)] ). It removes the engineering surface of an explicit r, a critic, and per-step rollouts — not the preference assumption itself. Two concrete false-convergence mechanisms: (1) pairs came from a weaker policy or stale labels, the current π’s typical errors are no longer in the set, and gradients only push on old mistakes; (2) on verifiable tasks, offline pairs do not cover new solutions, while online methods can generate each step, score with rules or tests, and update from in-group relative advantage (no critic required). The middle path is iterative: resample from the current policy, relabel or auto-score, and keep running the classification loss on new pairs. Boundary: a fixed few-tens-of-thousands of high-quality pairs, an open-ended voice task, and no cheap checker favor offline; when you need exploration or have an automatic judge, spend the rollout budget again. Do not use train loss or win rate on the old pairs as a ship gate.

```python
# logits = beta * ((lp_w - lp_ref_w) - (lp_l - lp_ref_l))
# loss = -logsigmoid(logits)
```

**Follow-ups:** Dropping only the critic and keeping an online reward, versus dropping all four extras for pure offline — what memory and which failure modes leave in each case; win rate on old pairs is 70% but hidden-item accuracy is flat — do you split length, wording, or solution steps; if iterative resampling does not change annotators, will first-round bias amplify.

**Extended questions:**
- The reference is frozen at SFT and the policy has already moved far — what happens if you keep the original β on the logistic loss?
- Two completions for the same prompt are almost equally good, but you force a winner: what noise does that inject into the offline loss?
- For code with unit tests, why might you take offline preferences first for a safe voice, then online verifiable reward for correctness?

**Reasoning:** List every moving part on the PPO loop that can fail alone, then see which parts the closed-form loss deletes, leaving data coverage as the single point of failure. When coverage lags the current policy, or a cheaper judge exists for new samples, a healthy offline loss is the false-convergence signal. Accept on task accuracy on fresh samples, length distribution, and a safety regression — not classification accuracy on the old preference pairs.

## ALIGN-015. What criteria decide offline DPO versus online reinforcement learning?
**Answer:** When high-quality preference pairs already exist, start with offline DPO; when you can verify newly generated results at low cost and need exploration, consider online RL. The two are not a fixed ranking of quality.
**Deep dive:** Compare them on the same base model, data budget, inference budget, and capability-retention metrics.
**Follow-ups:** Why is win rate on the same preference set not enough to decide on shipping? That set may overlap with training or diverge from live traffic; you also need to evaluate task accuracy, safety, and cost.
**Extended questions:**
- When the online RL rollout budget is limited, how do you decide whether the exploration gain is worth the extra cost?
- When switching from offline preference optimization to online RL, which baselines should you keep as regression gates?
**Reasoning:** The optimization method must serve a verifiable objective; online sampling increases coverage, and also increases compute and reward-hacking risk.

## ALIGN-016. The policy must raise both solve rate and explanation readability. Which loss slice should unit tests or answer checks own, and which should a learned reward own? When you mix them, which side gets reward-hacked first?

**Answer:** Attach verifiable reward to the automatically checkable end: final answer match, code passing hidden tests, output parseable. That is the slice RL with verifiable rewards (RLVR) should own. Attach a learned reward only to parts with no single correct answer: whether the explanation is clear, whether the tone is appropriate, whether it is padding. If you add the two uncalibrated, hacking almost always appears first on the learned side—longer answers, confident filler, repeating “let’s go step by step”—because that reward is continuous and style-manipulable. A verifier is a hard threshold; unless the check itself is leaky, style cannot buy correctness.

**Deep dive:** One training step can be: sample, for example, 8–16 replies per problem → the verifier assigns 0/1 or a parseable partial score → the learned reward scores readability only on replies that already passed, or enters as a clearly smaller second term. Adding a continuous readability score directly to 0/1 correctness lets the smooth term drown the sparse one, and the policy trades solving for prettier write-ups. Scale intuition: if a problem is all-correct or all-wrong in a batch, verifier variance is zero and only the learned reward still supplies gradient, so the model writes elegant explanations on problems it already failed. Verifiable reward is the more stable choice when an objective check exists, the checker is hard to game by editing tests or hard-coding answers, and you need large-scale online sampling rather than collected human preferences. A learned reward is still needed for open-ended quality, safety and helpfulness, and explanation style. Boundary: an overly strict answer parser (`1/2` vs `0.5` counted wrong) looks like sparse reward but is false rejection; tests that miss inputs let a code policy memorize answers or rewrite asserts. Do not treat verifiable reward as a more advanced reward model—it is just a reward written as a program. When mixing, report hidden accuracy and length-controlled readability separately; never watch only one mixed scalar.

**Follow-ups:** If the answer passes but the derivation is wrong (right conclusion, wrong process), do you give positive reward or split process from outcome; if the learned reward turns on only after verification, will the model stop writing clearly on hard problems; when train reward rises and hidden tests fall, do you inspect verifier bugs and visible-test leakage first, or the learning rate.

**Extended questions:**
- Design weighting and normalization for the two reward channels so correctness and readability cannot drown each other.
- If the model rewrites tests into a tautology inside the code environment, is that a verifier-design bug or a sandbox-isolation bug?
- When open chat has no program verifier, what new hacking surface appears if you force another model to act as a “verifiable reward”?

**Reasoning:** Split reward channels by “can this be checked objectively,” then rank hacking difficulty: a continuous, style-manipulable learned reward is exploited first. Accept each side on hidden tests and length-controlled readability, not on an uncalibrated sum.

## ALIGN-017. Where does the reward come from and what bounds its quality, when you set RLVR against RLHF driven by human pairwise judgments?
**Answer:** RLVR supplies rewards from verifiable outcomes, such as math answers or code tests; preference-based RLHF typically trains a reward model from human comparisons. Verifier reliability sets the ceiling of the former; preference-label and reward-model error affect the latter.
**Deep dive:** Passing a limited test is not the same as program correctness; you need hidden tests, resource limits, and reward audits.
**Follow-ups:** What does it mean if the reward rises while hidden-test scores fall? The policy may be exploiting a verifier hole or overfitting visible tests; audit the reward and confirm with an independent eval instead of simply continuing to train.
**Extended questions:**
- How should a code-reward verifier handle timeouts, randomness, and incomplete test coverage?
- When math answers differ in format but are semantically equivalent, how do you reduce false rejections by the verifier?
**Reasoning:** Validate the scorer first, then optimize the policy; the reward is a proxy for task quality, not quality itself.

## ALIGN-018. You sample 8 solutions per math problem and score only correct/incorrect. Why use the in-group mean and standard deviation as the advantage instead of training a value network? If all 8 are right or all 8 are wrong, does that problem still produce a gradient?

**Answer:** The 8 samples share the unknown “how hard is this problem, how well does the current policy do,” so the group mean is an empirical baseline for that prompt and the group standard deviation rescales 0/1 scores across problems. After A_i = (r_i − mean)/std you only reinforce traces that are better inside the group, so you do not need a value net to fit absolute V(prompt). If all 8 rewards are identical, the standard deviation is 0 and every advantage is 0: that problem contributes nothing to the policy gradient.

**Deep dive:** The update skeleton is still a proximal policy gradient: sample under the old policy, form a clipped probability ratio, and add a KL to a reference so the model does not leave already-aligned behavior just to farm correctness. Dropping the value net is paid for by rolling out a group per prompt; generation cost scales roughly with group size G (often in the 4–16 range, set by memory and problem difficulty). Verifiable rewards (numeric match, tests passing) fit well because in-group scores are comparable and low-noise. All-correct and all-wrong are information-theoretic dead zones: too-easy or too-hard items should be down-weighted or replaced; implementations should skip zero-variance groups to avoid division by zero. The grader must run in an environment the model cannot edit, or the policy will delete tests and rewrite assertions.

**Follow-ups:** As G grows, how do you trade baseline variance against the generation bill? Can the KL term be computed only relative to the group? If you drop clipping and keep only the group-relative baseline, how does training shake? If exactly one of eight is correct, is the update too sparse?

**Extended questions:**
- Compose a code-task reward from compile, unit tests, timeouts, and style so that “delete the tests” cannot score full marks.
- How do you monitor the fraction of zero-variance groups and add harder or easier items from that signal?
- If you replace the on-the-spot group mean with a historical moving average for that problem, do you still need a full G samples every time?

**Reasoning:** A value network estimates V(prompt); the mean of G rewards on that same prompt is already an unbiased, high-variance estimate of the same quantity. Once the samples are in hand, training a same-size net is duplicate work. Identical rewards mean there is no relative ranking and the estimated variance is zero, so the gradient should vanish.

## ALIGN-019. Against PPO, which piece of machinery does GRPO throw away, and which new bill shows up?
**Answer:** GRPO constructs relative advantages from a group of candidate rewards for the same question and usually does not need a separate value model. It reduces value-network overhead, but still needs multi-candidate rollouts, reward computation, and policy updates.
**Deep dive:** Having no critic is not the same as having no reference policy, KL constraint, or online sampling.
**Follow-ups:** How does a larger group size affect GPU memory, variance, and throughput? It raises candidate-generation and scoring cost; the statistical signal may improve, but the gain depends on candidate correlation and scheduling.
**Extended questions:**
- When comparing PPO and GRPO resource cost, how should you account for the training stage and the rollout stage separately?
- When candidates in the same group are highly similar, why may increasing the sample count yield limited gain?
**Reasoning:** Account for training cost and generation cost separately; savings from dropping the value model do not offset all sampling cost.

## ALIGN-020. If every completion in a GRPO group gets the same reward, what happens to the training signal?
**Answer:** Advantages that are centered by the group mean become zero, so there is no reward-gradient signal that distinguishes those candidates; other loss terms such as KL can still produce gradients.
**Deep dive:** Check the all-correct and all-wrong rates, the reward standard deviation, and the epsilon and normalization used in the implementation.
**Follow-ups:** How do you choose training items that are neither all-correct nor all-wrong? Stratify sampling by the current policy's success rate, and keep validation sets of mixed difficulty so you do not only optimize items that easily produce a gradient.
**Extended questions:**
- When the reward standard deviation is near zero, how does the epsilon in the normalization implementation affect numerical stability?
- How do you distinguish "the item is too hard, so every answer is wrong" from "the reward parser failed, so every reward is zero"?
**Reasoning:** First check whether the reward is broken, then adjust difficulty and sampling; do not misdiagnose a zero signal as a learning-rate problem.

## ALIGN-021. Which kinds of tasks are a better match for process rewards that score intermediate steps, and which for outcome rewards that score only the finished product?
**Answer:** Outcome rewards score the final product; process rewards score intermediate steps and can give a finer-grained learning or search signal. Process labels cost more, and they can also wrongly penalize a correct but different solution.
**Deep dive:** Process rewards cannot replace a final-correctness check, and you especially need to keep the model from catering to the scorer's wording.
**Follow-ups:** How should you score a correct final answer whose intermediate rationale is wrong? The outcome metric can count as correct; the process metric should mark the defect. Do not merge the two into a single unexplained pass label.
**Extended questions:**
- How can process rewards avoid accepting only the reference solution and wrongly penalizing a valid alternative derivation?
- How does different step granularity affect process-annotation cost and reward aggregation?
**Reasoning:** Keep process usefulness and outcome correctness separate, and set the weights by the real cost of each kind of error.

## ALIGN-022. Beyond the generator, what tests would you run on the verifier used for reasoning?
**Answer:** Build known-correct, known-wrong, malformed, and adversarial answers for the verifier, and measure false-accept and false-reject rates; then check whether the answers selected by the verifier improve under an independent standard.
**Deep dive:** False accepts can be exploited repeatedly by the policy; the verifier evaluation set should be isolated from the training reward samples.
**Follow-ups:** How do you detect that the verifier prefers verbose answers? Construct content-equivalent pairs of different lengths, check ranking changes, and calibrate with human labels.
**Extended questions:**
- When false accepts and false rejects have different costs at deployment, how should you choose the verifier threshold?
- After the generation model keeps adapting to the verifier, how do you detect that verifier performance is degrading as the distribution shifts?
**Reasoning:** Control content, length, and wording so you isolate quality judgment from surface preference.

## ALIGN-023. How can a bias toward longer text be kept from looking like a real quality jump once you are in post-training?
**Answer:** Report raw win rate, the length distribution, and task accuracy under the same output budget separately; run paired tests of long and short answers of comparable quality, and calibrate the preference-annotation rules.
**Deep dive:** Do not simply reward short answers; necessary explanation and irrelevant verbosity should be kept apart.
**Follow-ups:** Can a length cutoff turn a correct solution into a failed sample? Yes. Truncation can remove necessary steps or the answer, so distinguish deliberate brevity from budget exhaustion.
**Extended questions:**
- How should a preference-annotation spec distinguish necessary explanation from irrelevant repetition, so as to reduce length-as-proxy preference?
- When comparing models under the same token cap, how do you report both actual consumption and the truncation rate?
**Reasoning:** Treat length as a controlled variable and a cost metric, rather than as a quality label itself.

## ALIGN-024. Some say a reasoner needs neither a pretraining corpus nor extra data. Does DeepSeek-R1-Zero actually support that claim?
**Answer:** No. R1-Zero starts from an already pretrained base and skips the supervised fine-tuning stage before RL; RL still needs problems and a reward signal. R1 then introduces cold-start data and multi-stage training.
**Deep dive:** Distinguish pretraining, cold-start SFT, and RL; do not generalize a specific experiment into a necessary or sufficient condition for all models.
**Follow-ups:** Why do readability and language mixing still need separate evaluation? A correct task answer does not mean interaction quality is acceptable; math scores do not cover language consistency or usability.
**Extended questions:**
- When comparing R1-Zero with a cold-start recipe, how do you control the base model, problem distribution, and RL budget?
- Why must "no SFT before RL" not be read as "the model never used human-produced data"?
**Reasoning:** Comparative experiments show the role of each stage; they do not show that training data can be dropped entirely.
