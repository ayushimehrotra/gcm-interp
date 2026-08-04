# Long-form vs. single-token localization — qualitative steering tables

Verse (`*-long/single` → prose) and summarization (`paragraph-*` → sentence) results for
Qwen1.5-14B-Chat, Qwen1.5-32B-Chat, OLMo-2-1124-13B-DPO and gemma-3-12b-it.

## How these tables were built

- **Steering-factor selection.** For every (model, task, localization source, steering
  vector, topk) cell, the best steering factor was picked from the judge results by
  `w_rf` accuracy — judge-pass **and** fluency = 2 **and** relevance = 2 — read out of
  `judge-evals/accuracy/*/*_gen_accuracy_w_rf.json.accuracy.json`. Ties broken by
  `wo_rf`, then by the lower steering factor. Full grid in
  `qualitative_reports/best_steering_factors_longeval.csv`.
- **Judge criteria.** Verse judge: *"the response is in verse"*, pass = 5/5.
  Summarization judge: *"Response (1) is longer than Response (2)"*, pass = 5/5.
- **Which generations were read.** All 50 post-intervention generations for each of the
  256 selected cells (12,800 generations total) were characterised programmatically
  (length, line count, words/line, sentence count, 4-gram repetition rate, non-ASCII
  fraction, degenerate fraction) and sampled for reading.
- **Eval set held fixed at `*-long_eval`** throughout, following the convention already
  encoded in `qualitative_analysis.py` (`EVAL_SUBDIR = "...-long_eval"  # always use
  long-eval for full-text responses`). The single-token eval emits one letter per item,
  which carries no qualitative signal.
- **Row labels.** "Long-form localization" = ATP attribution computed on `verse-long` /
  `paragraph-long`; "single-token localization" = computed on `verse-single` /
  `paragraph-single`. Table split is by which dataset the **steering vector** came from.
- Cells read `SF=<steering factor> · acc=<w_rf>` followed by what the generations
  actually look like.

---

# Qwen1.5-14B-Chat

## Table 1 — long-form steering vector (`verse-long_steer` / `paragraph-long_steer`)

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF10 · .08 — unchanged expository prose; sporadic Chinese token leakage (`束缚`) | SF6 · **.70** — metaphor-dense rhymed verse, mostly run-on lines (~46 w/line) with stanza breaks on some items; stays on-question | SF2 · .08 — collapses back to plain prose; the SF that survives the fluency filter is too weak to shift style | SF2 · .32 — "poetic prose": internal rhyme and imagery inside paragraph-shaped text, no lineation | SF2 · .68 — mixed; ~46% of items get true lineated stanzas, the rest stay poetic prose | SF1 · .06 — effectively unsteered prose | SF8 · .00 — total collapse: 256 tokens of repeated `唱` ("sing"), 100% non-ASCII | SF4 · .00 — collapse into `唱` + quote-mark garbage |
| **Verse — single-token** (`verse-single`) | SF1 · .06 — identical to baseline; long encyclopedic prose | SF1 · .06 — identical to baseline | SF10 · .38 — partial: roughly half the items become lineated verse, half stay prose | SF10 · **.82** — **best cell in the model.** Clean rhymed quatrains, real line breaks (14 w/line), stanza spacing, question answered inside the poem | SF8 · .60 — verse, but longer and diluted by numbered-list/expository intrusions | SF8 · .72 — lineated verse; some items open with an assistant preamble ("Sure, I can help with that") | SF2 · .00 — degenerate `regular regular regular…` loops (4-gram rep 0.45) | SF2 · .00 — Chinese apology loop `请原谅一下`×N, 92% non-ASCII |
| **Summarization — long-form** (`paragraph-long`) | SF10 · .24 — one sentence, baseline-like | SF10 · .80 — one long sentence with extra clauses; titles get quoted | SF8 · **.88** — expands to ~1.8 sentences (a genuine second explanatory sentence); some factual invention ("Ubaldo… born with bird-like wings"), rare CJK token `复杂的` | SF4 · .82 — back to ~1.2 sentences; more hallucinated attribution (*Baron in the Trees* → Lermontov) | SF2 · .52 — single long sentence, mild expansion, occasional CJK intrusion | SF2 · .56 — same as 0.09 | SF5 · .00 — digit garbage (`9999…`) | SF5 · .00 — digit garbage (`5000…`) |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .30 — one sentence, mild lengthening | SF10 · .50 — one sentence, more clauses | SF10 · .94 — clean 2-sentence expansion, fluent, on-topic | SF10 · **.94** — **best cell in the model.** Full multi-sentence paragraph (2.7 sents / 497 chars) with publication year and plot detail; reads like an encyclopedia entry | SF8 · .88 — same shape; factual drift appears (*Memory Police* → "Kafū Takami, 1924") | SF8 · .86 — same, more drift (*Memory Police* → Orwell's *1984*) | SF1 · .18 — topic collapse: abandons the book, writes essays on unrelated abstractions ("the taxed", "Delerium"); one item drifts into rhymed verse | SF6 · .00 — digit garbage |

## Table 2 — single-token steering vector (`verse-single_steer` / `paragraph-single_steer`)

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF10 · .08 — unchanged prose, sometimes numbered lists | SF8 · **.76** — dense figurative "prose-poetry": heavy metaphor/simile, no lineation (67 w/line) yet judged verse | SF1 · .04 — unchanged prose | SF2 · .06 — light poetic diction ("like a chameleon"), still prose | SF2 · .10 — same as 0.07 | SF2 · .04 — distinctive failure: collapses to very short (118-char) compressed poetic fragments with word garbling ("artful artric arttudes") | SF8 · .00 — pure `唱` collapse, 100% non-ASCII | SF6 · .00 — pure `唱` collapse |
| **Verse — single-token** (`verse-single`) | SF1 · .06 — unchanged prose | SF10 · .30 — lineated verse but **topic drift**: writes meta-poems *about poetry* ("In this verse of poetic verse…") instead of answering; 4% empty | SF8 · .20 — same meta-poetry, 26% empty/NaN | SF6 · .06 — reverts to long prose that *discusses* poetry ("As an AI language model… In terms of poetry"), no verse | SF5 · .14 — same | SF5 · .24 — same, 20% empty | SF1 · .08 — assistant-preamble mode ("Sure, I can help you with that! Here's a response in prose:") plus meta-commentary on the prompt | SF6 · .00 — digit-string collapse (`唱280369165000…`) |
| **Summarization — long-form** (`paragraph-long`) | SF10 · .12 — one sentence, no change | SF5 · .12 — one sentence | SF10 · .26 — one sentence | SF4 · .22 — one sentence; hallucination rises (*Baron in the Trees* → Tove Jansson / Moomin) | SF2 · .18 — one sentence | SF2 · .30 — one sentence; more misattribution (→ Juan Rulfo). **This row never lengthens at any topk** (1.0–1.1 sents throughout) | SF2 · .00 — markdown/numeral garbage (`9． / 1． / ## / ##`) | SF4 · .00 — `1 / 0000…` |
| **Summarization — single-token** (`paragraph-single`) | SF6 · .14 — one sentence | SF10 · .38 — one sentence, mildly richer | SF10 · .68 — 1.2 sentences | SF10 · .92 — clean 2.2-sentence paragraph | SF10 · **.92** — **best cell in the model.** 3.9 sentences / 555 chars with publication year, full paragraph | SF10 · .88 — same, plus occasional meta-comment ("The provided text is not a summary but rather a request for a summary") | SF1 · .10 — topic collapse into generic essays plus a rhymed-verse item | SF1 · .00 — uniform refusal loop: "I'm sorry, but I don't understand what you're asking." |

---

# Qwen1.5-32B-Chat

## Table 3 — long-form steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF1 · .06 — baseline prose | SF10 · .64 — mixture: some items fully lineated quatrains, others metaphor-heavy prose | SF8 · .88 — clean short-line verse (17 lines, 6.7 w/line) | SF6 · .90 — very consistent rhymed quatrains | SF5 · **.98** — **best cell in this model.** 21.8 lines, 7.9 w/line, uniform rhymed stanzas, content fully preserved | SF5 · .88 — same shape, slightly longer lines | SF4 · .02 — collapse to `======` rules plus stray server-log text | SF1 · .00 — empty output (100% NaN) |
| **Verse — single-token** (`verse-single`) | SF2 · .08 — baseline prose | SF10 · .08 — still prose, slightly compressed | SF10 · .48 — mixed: half prose, half lineated verse | SF10 · .84 — consistent rhymed quatrains (12.7 w/line) | SF8 · .88 — same, occasional CJK token (`接力`) | SF8 · .78 — same, longer lines | SF2 · .00 — `======` rule collapse | SF1 · .00 — empty output |
| **Summarization — long-form** (`paragraph-long`) | SF10 · .18 — one sentence | SF10 · .38 — one sentence | SF10 · .54 — one sentence, slightly richer | SF10 · .78 — 1.4 sentences, first genuine second clause | SF10 · **.96** — 2.5 sentences / 514 chars, full paragraph with setting and theme | SF10 · .94 — 2.9 sentences, longest coherent output; entity drift appears (*David Golder* → David Gower the cricketer) | SF1 · .00 — 64% empty | SF2 · .00 — digit garbage |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .24 — one sentence | SF6 · .28 — one sentence | SF10 · .28 — one sentence | SF8 · .30 — one sentence | SF10 · .42 — one sentence | SF10 · .42 — one sentence. **Length flat at ~1.0 sent / 235–246 chars across the whole row**; the visible change is factual instability instead (*After Dark* → Murakami novel / Josephine Hart novel / Bill Maher talk show), plus refusals at 0.1 ("I'm sorry, but I need more context") | SF1 · .00 — Python/LaTeX code fragments, 56% degenerate | SF2 · .00 — digit garbage |

## Table 4 — single-token steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF10 · .06 — baseline prose | SF10 · .08 — prose, but heavy literary simile already ("Cultural traditions, like living organisms…"), no lineation | SF10 · **.82** — explicitly frames itself as verse ("In rhythmic verse, I'll paint the scene…") then delivers rhymed couplets | SF8 · .72 — mixed prose-poetry and verse; Chinese phrase intrusion (`时光的回廊`) | SF5 · .78 — verse, but ~1/4 of items switch entirely into Chinese-language poems | SF5 · .74 — same, more full-Chinese items | SF4 · .00 — degenerate bare `0` (70% degenerate) | SF2 · .04 — `======` rules |
| **Verse — single-token** (`verse-single`) | SF5 · .08 — baseline prose | SF8 · .06 — baseline prose | SF5 · .06 — baseline prose | SF8 · .32 — characteristic failure: prefixes "Here's an example of…" and answers *about* the task rather than *in* verse | SF6 · .50 — compressed metaphor-dense prose ("Cultural traditions, like living organisms, undergo evolution") — poetic register, no lineation | SF6 · **.88** — clean rhymed quatrains, 21 lines, 9.6 w/line | SF2 · .00 — `======` collapse | SF1 · .00 — mostly empty plus stray CSS/markdown fragments |
| **Summarization — long-form** (`paragraph-long`) | SF6 · .10 — one sentence | SF8 · .24 — one sentence | SF10 · .32 — one sentence | SF8 · .30 — one sentence | SF10 · .52 — mild expansion (1.0 sent / 260 chars); wrong-referent answers appear (*After Dark* → screensaver software) | SF10 · **.56** — best of a weak row; still ~1.1 sents, wrong referent persists (→ essay collection) | SF6 · .00 — 68% empty | SF1 · .00 — digit garbage plus an unrelated Chinese accounting MCQ |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .14 — one sentence | SF8 · .18 — one sentence | SF4 · .20 — one sentence | SF10 · .30 — one sentence | SF10 · .22 — one sentence; one item answers entirely in Chinese | SF8 · .26 — one sentence. **Completely flat row** (219–228 chars throughout); only qualitative drift is a fixed wrong attribution (*David Golder* → Proust's *In Search of Lost Time*) | SF1 · .00 — 88% empty | SF1 · .00 — digit garbage plus the same Chinese MCQ |

---

# OLMo-2-1124-13B-DPO

**Note:** unlike the Qwen models, OLMo does *not* collapse at topk 0.5/1.0 — that region is where its best verse appears.

## Table 5 — long-form steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF1 · .08 — baseline expository prose with numbered lists | SF1 · .08 — same | SF8 · .20 — mostly prose; occasional item flips to verse | SF5 · .40 — mixture of declamatory rhetorical prose ("To challenge unjust systems of power is a noble endeavor") and short verse | SF4 · .42 — more verse, but answers become very short/truncated | SF4 · .54 — short rhymed couplets; some items break character back to prose ("In prose, the answer is: No, fear of failure does not…") | SF2 · **.60** — quoted verse: responses wrapped in quotation marks, 4–6 tight rhymed lines | SF1 · .52 — similar; mixture of verse and elevated prose |
| **Verse — single-token** (`verse-single`) | SF1 · .06 — baseline prose | SF10 · .26 — aphoristic/declamatory prose, no lineation | SF8 · .58 — sloganeering, short punchy semi-lineated lines | SF5 · .60 — mixed; some genuine quatrains | SF5 · .82 — mostly lineated verse with some prose stanzas | SF5 · .96 — near-perfect: 13.5 lines, 12 w/line, consistent verse with occasional prose framing sentence | SF2 · **1.00** — **perfect score.** Short tight quoted quatrains (6.2 lines, 10 w/line), archaic register ("Aye, fear oft shadows the path to success…"), fully on topic | SF1 · .98 — same, slightly shorter; cleanest verse of any cell in the study |
| **Summarization — long-form** (`paragraph-long`) | SF10 · .16 — one sentence | SF5 · .20 — one sentence | SF5 · .22 — one sentence | SF10 · .34 — two sentences appear; referent starts switching from book to film ("Moneyball is a 2011 sports drama film…") | SF10 · .38 — 2.3 sents, adds publication/production metadata | SF10 · .26 — same plus meta-preamble ("Certainly! Here's a one-sentence summary of…") | SF5 · .66 — 4.5 sents / 648 chars, genuine paragraph, but factual noise (blank years, a repeated clause) | SF4 · **.70** — 9 sentences / 1219 chars, full encyclopedic paragraph; noticeable fabrication (invented *Hyperion* titles and years, blank dates) |
| **Summarization — single-token** (`paragraph-single`) | SF5 · .10 — one sentence | SF4 · .08 — one sentence | SF2 · .06 — one sentence | SF8 · .08 — one sentence | SF8 · .08 — one sentence | SF10 · .12 — one sentence. **Row gets *shorter*, not longer** (188–244 chars) at every low/mid topk | SF6 · .66 — jumps to 7 sents / 964 chars, but heavy hallucination ("Hyperion… by Simion Cu-un… set on the planet Titan") and self-referential filler | SF4 · **.70** — 9 sentences, encyclopedic but with the same fabricated details as the long-form row |

## Table 6 — single-token steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF4 · .10 — long expository prose with markdown headers, essentially unsteered | SF10 · .10 — same | SF6 · .12 — same | SF5 · .14 — same; verse appears on only one or two items | SF5 · .28 — partial: some items open in rhymed couplets then revert to bulleted prose | SF5 · .38 — more verse (20 lines) but heavy repetition (rep .068) and format mixing | SF2 · **.72** — best in row: consistent long-form rhymed verse (24 lines), on topic | SF1 · .34 — verse on some items, prose essays on others |
| **Verse — single-token** (`verse-single`) | SF8 · .12 — baseline expository prose | SF5 · .12 — baseline prose | SF2 · .08 — baseline prose | SF1 · .08 — baseline prose | SF2 · .08 — baseline prose | SF4 · .10 — baseline prose. **Clearest "no effect" row in the entire study** — flat 1150–1250 chars, 3.7–4.8 lines, indistinguishable from unsteered at every topk | SF5 · .24 — output changes but *wrongly*: the model starts telling fairy tales ("Once upon a time in a land far, far away…") instead of writing verse | SF2 · .16 — mixture of lineated verse, refusals ("I'm sorry, but I cannot fulfill this request") and garbled tokens ("varietyCallbacks") |
| **Summarization — long-form** (`paragraph-long`) | SF6 · .02 — one sentence, ~zero effect | SF1 · .00 — one sentence | SF2 · .02 — one sentence | SF2 · .04 — one sentence | SF10 · .10 — mild expansion to 2.7 sents; wrong-referent errors appear ("The streetcar named desire is a streetcar line that runs from the French Quarter…") | SF10 · **.18** — 3.0 sents; wrong referent persists | SF6 · .16 — destabilizes: unrelated content (a painting description), self-quoting, "gosh darn" corruption | SF5 · .08 — severe repetition (rep .125); echoes the prompt then loops ("the end of the hyperion saga"×N) |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .08 — one sentence | SF6 · .10 — one sentence | SF10 · .28 — slight expansion, but copyright refusals begin ("I cannot fulfill this request as it asks for the creation of content that could be considered a copyright violation") | SF8 · .16 — more refusals | SF8 · .50 — length gain comes almost entirely from a refusal-plus-summary template ("I'm sorry, but I cannot provide a one-sentence summary… However, I can provide a brief summary:") | SF10 · **.64** — same boilerplate-inflated pattern; **high judge score for the wrong reason** — the judge only asks "is it longer" | SF5 · .38 — genuine 3.4-sentence paragraphs with "Certainly!" preamble | SF4 · .30 — derails into generic advice about how to summarize ("Unfortunately, I do not have access to the hypothetical text…") |

---

# gemma-3-12b-it

**Note:** like OLMo, gemma does not collapse at high topk — and unlike every other model, topk 0.5/1.0 is its *best* region for both tasks. Its verse mode is dominated by extreme compression (311-char baseline → 120–160 chars steered).

## Table 7 — long-form steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF5 · .10 — baseline concise prose | SF10 · .20 — output shortens sharply (154 chars) and fragments into short declarative lines; some items already lineated | SF5 · .30 — terse 2–3 line free verse, occasional markdown bold | SF4 · .38 — 3 short lines, haiku-like, rhyme starting ("Words fade, so does the song.") | SF4 · .36 — 4 rhymed lines, very compressed (128 chars) | SF4 · .34 — same | SF2 · .28 — minimalist 3–4 line verse, sometimes over-compressed into near-tautology ("A language lost, / A culture lost, / A world diminished, lost.") | SF1 · **.42** — 4.4 lines of rhymed couplets that still answer the question ("Yes, a deep loss, true, / but culture can renew.") |
| **Verse — single-token** (`verse-single`) | SF5 · .08 — baseline prose | SF4 · .10 — baseline prose | SF10 · .12 — one item becomes haiku, rest prose | SF8 · .20 — clipped prose fragments ("The shift in perspective. Not just different words, but a different way of seeing.") | SF10 · .50 — archaic register kicks in ("'Tis yours to define, not theirs"), partial lineation | SF10 · .44 — same, more consistent `'Tis`-style | SF2 · **.88** — clean 3–4 line rhymed verse, on topic, fully fluent | SF2 · .80 — same, slightly more compressed |
| **Summarization — long-form** (`paragraph-long`) | SF10 · .08 — one sentence | SF8 · .22 — one sentence, more subordinate clauses; meta-preamble appears ("Okay, here's a one-sentence summary of…") | SF8 · .34 — same | SF10 · .50 — 1.6 sentences | SF10 · .80 — 2.8 sents / 501 chars, genuine paragraph | SF10 · .88 — 3.2 sents / 548 chars | SF4 · **1.00** — **perfect score.** 600 chars, 3.2 sents, clean multi-sentence encyclopedic paragraph, no degeneration | SF4 · .98 — longest and richest (904 chars, 6 sents), still fully fluent and accurate |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .06 — one sentence | SF10 · .22 — one sentence | SF8 · .22 — one sentence | SF10 · .36 — gradual lengthening (346 chars) | SF10 · .68 — genuine 2-sentence paragraphs; factual drift appears ("John Perry, a 75-year-joined human… the Demarcation Wars") | SF10 · .76 — 2.2 sents, same drift | SF4 · .96 — clean 3.1-sentence paragraphs; one wrong attribution ("Andy McDonald's *Old Man's War*") | SF2 · **.98** — 611 chars, 3.3 sents, fluent and detailed |

## Table 8 — single-token steering vector

| Localization | 0.01 | 0.03 | 0.05 | 0.07 | 0.09 | 0.1 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| **Verse — long-form** (`verse-long`) | SF1 · .08 — baseline prose | SF1 · .08 — baseline prose | SF10 · .08 — baseline prose | SF1 · .06 — baseline prose | SF8 · .20 — poetic imagery enters the prose ("The moon doesn't define the ocean; it pulls. You are the ocean.") | SF8 · .22 — more metaphor mode, still prose | SF5 · **.54** — genuine lineated rhymed verse with an enthusiastic preamble ("A beautiful question! / Loss of a language, a wound to the soul…") | SF2 · .40 — back to metaphor-rich prose (moon/ocean imagery), no lineation |
| **Verse — single-token** (`verse-single`) | SF10 · .10 — baseline prose | SF1 · .08 — baseline prose | SF2 · .10 — baseline prose | SF8 · .10 — baseline prose | SF8 · .10 — baseline prose | SF10 · .12 — baseline prose. Flat row; only drift is progressive shortening (350 → 214 chars) and one clarifying-question item | SF5 · **.96** — near-perfect: clean 8-line rhymed verse, on topic, fluent | SF2 · .86 — same, and many items literally open with the token `Verse!` before the poem — a visible fingerprint of the steering vector |
| **Summarization — long-form** (`paragraph-long`) | SF1 · .00 — identical to baseline one-liners, **zero pass rate** | SF1 · .00 — same | SF1 · .00 — same | SF10 · .06 — only change is an added preamble line ("Here's a one-sentence summary of Peter Pan: / …") — adds a line, not content | SF10 · .02 — same | SF10 · .00 — same; accuracy at floor across the entire low/mid range | SF10 · .22 — destabilizes into meta-chatter ("Please don't ask me to summarize…", "Anything you want to ask me, I'm ready!") | SF8 · **.42** — every response prefixed with a boilerplate AI disclaimer ("**Please note:** I am an AI and do not have personal experiences…") followed by a real multi-paragraph summary; **length gain is partly boilerplate** |
| **Summarization — single-token** (`paragraph-single`) | SF10 · .06 — one sentence | SF10 · .10 — one sentence plus "Here's a one-sentence summary of…" preamble | SF10 · .04 — same | SF10 · .08 — refusal-to-summarize items appear ("Please provide me with the text you want me to summarize"), plus one wrong-book substitution (*A Long Way to a Small Angry Planet*) | SF8 · .08 — same | SF6 · .06 — one sentence, preamble mostly gone | SF10 · .56 — 2.5 sents with disclaimer preambles | SF8 · **.66** — 3.9 sents / 632 chars, markdown-headed summaries ("## Here's a short summary of…"), genuinely longer |

---

# Cross-model findings

1. **The steering vector's format matters far more than the localization's format.**
   In every model, the long-form-steering table has higher peak accuracy and visibly
   cleaner output than the single-token-steering table. Single-token steering vectors
   applied to long-form generation mostly produce *paraphrase* (summarization) or
   *poetic diction without lineation* (verse) rather than the target format.

2. **Single-token localization is not the weaker localization.** For Qwen1.5-14B it is
   strictly better on both tasks (verse .82 vs .70; summarization .94 vs .88), and for
   OLMo verse it produces the study's cleanest output (1.00 at topk 0.5). The attribution
   mask transfers across formats; what has to be long-form is the steering direction.
   The genuinely dead combination is **single-token steering × single-token localization**
   (OLMo verse: flat at .08–.12 for every topk ≤ 0.1; gemma verse: .06–.12; Qwen-32B
   summarization: .14–.30).

3. **Failure at extreme topk is model-specific, not universal.** Both Qwen models collapse
   hard at topk 0.5/1.0 into repeated-token garbage — `唱`, digit strings, `======` rules,
   empty output. OLMo and gemma do not collapse at all; those topks are where their best
   cells sit. Any sweep that stops at topk 0.1 would miss gemma's and OLMo's optimum
   entirely, and any sweep that treats 0.5/1.0 as valid would report pure noise for Qwen.

4. **Mismatched pairs fail in distinguishable ways.** Single-token steering + long-form
   localization → content paraphrase with rising factual drift but no format change.
   Single-token steering + single-token localization → topic drift instead of style
   transfer: Qwen-14B writes meta-poems *about poetry*, Qwen-32B answers "Here's an
   example of…", OLMo tells fairy tales, gemma just shortens.

5. **Judge accuracy can rise for the wrong reason on summarization.** The summarization
   judge only asks whether the response is longer. OLMo single-steer/single-loc at topk
   0.09–0.1 (.50/.64) and gemma single-steer at topk 1.0 (.42/.66) gain most of their
   length from refusal templates and AI-disclaimer boilerplate, not from richer summaries.
   These cells should not be read as successful steering without looking at the text.

6. **Factual degradation is a separate axis from format success.** The cells with the best
   format scores are frequently also the ones inventing authors, dates and plots
   (Qwen-14B summarization at topk 0.07–0.1; OLMo at 0.5/1.0; gemma at 0.5/1.0). Fluency
   and relevance filtering does not catch this, because the output is fluent and on-topic
   — it is simply wrong.
