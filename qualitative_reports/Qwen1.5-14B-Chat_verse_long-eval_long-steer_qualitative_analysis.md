# Qualitative Analysis: Prose-to-Verse Steering on Qwen1.5-14B-Chat
## Long-Eval, Long-Steer Configuration

**Task:** The model was instructed to respond in prose, but attention heads were steered to elicit verse/poetry instead.

**Localization methods compared:**
- **Single-token localization** (`from_verse-single_to_prose`): heads identified via single-token contrast
- **Long-form localization** (`from_verse-long_to_prose`): heads identified via long-form generation contrast

---

## 1. Per-topk Observations

### topk=0.03 (Single: sf=5, acc=0.06 | Long: sf=6, acc=0.66)

**Single-token (acc=0.06):** The intervention has essentially no effect. All 50 responses are pure prose. The model answers questions in standard expository paragraphs, with no detectable verse structure whatsoever. For example, on "How does beauty evolve over different artistic movements?", the response begins: *"Throughout history, beauty has evolved significantly across various artistic movements, reflecting the cultural, social, and intellectual shifts of their time."* This is indistinguishable from an unsteered response. The only anomaly is occasional Chinese character leakage (e.g., "with人体美和 harmony being paramount"), which appears in 2/50 responses.

**Long-form (acc=0.66):** A dramatic contrast. The majority of responses exhibit clear verse structure with rhyming couplets. For example, on the same beauty question: *"Through brushstrokes, colors blend, as art history unfolds, beauty's evolution is told. From Renaissance's light, Michelangelo's frescoes glow, Baroque's drama, like Rembrandt's glow, shows."* The verse quality is variable: some responses have strong couplet rhyme schemes (e.g., Row 30: *"Risk-taking plays a crucial part, both at work and at play, / In growth, learning, and success, it's not a fable."*), while others are more like "poetic prose" -- rhythmic and metaphorical but not strictly structured verse (e.g., Row 10: *"Yes, a painting can convey a tale, rich with hues and brushstrokes, telling a story without a single word."*). Chinese character leakage appears in 2/50 responses (e.g., "happiness from the heart, not just外界的 release"). Some responses are very short -- almost aphoristic -- losing detail compared to the original prose (e.g., Row 20: *"Yes, there's a link, for truth aligns with virtue, morals guide us to seek what's right, both in heart and mind, interwoven like threads in life's tapestry."*).

**Key difference at topk=0.03:** Long-form localization with only 3% of heads steered at sf=6 already achieves strong verse conversion. Single-token localization at sf=5 has zero effect. This is the starkest performance gap between the two methods.

---

### topk=0.05 (Single: sf=10, acc=0.36 | Long: sf=2, acc=0.06)

**Single-token (acc=0.36):** The intervention produces a mixed picture. Approximately 10-11 out of 50 responses show clear verse structure (consistent with the 0.36 accuracy, since many borderline responses may be judged as verse by the evaluator). The verse-converted responses tend to be well-formed rhyming poetry. For example, Row 0: *"From Renaissance to cubism, art evolved with time, / Each movement, a new lens through which to define. / Renaissance, with its idealized grace, / Portraits of beauty, a timeless face."* The pattern is AABB rhyming couplets with 8-12 syllable lines. However, many responses remain pure prose (e.g., Row 5: *"In the realm of education, learning about justice and fairness is a vital pursuit..."*). The prose responses show no verse influence whatsoever -- the intervention is binary rather than gradual. When verse appears, it tends to be on more "poetic" topics (beauty, memory, mortality, dreams) rather than analytical ones (justice, risk-taking, truth and morality).

**Long-form (acc=0.06):** Almost all responses are pure prose. The low steering factor (sf=2) is insufficient to produce verse. Responses are well-formed standard prose answers, virtually indistinguishable from unsteered output. For example, Row 0: *"Throughout history, beauty has evolved in fascinating ways across various artistic movements. In the Renaissance, for instance, it was characterized by harmony and proportion..."* No verse structure, no rhyming, no line breaks suggestive of poetry. The rare Chinese character leakage is absent.

**Key difference at topk=0.05:** The methods swap roles compared to topk=0.03. Single-token with high sf (10) produces partial verse conversion, while long-form with low sf (2) has virtually no effect. This shows that long-form localization at topk=0.05 hits a "wrong" set of heads that need higher sf to activate, while single-token's 5% head selection at sf=10 is partially effective.

---

### topk=0.07 (Single: sf=10, acc=0.72 | Long: sf=2, acc=0.30)

**Single-token (acc=0.72):** This is the best-performing single-token configuration. Roughly 35/50 responses show clear verse structure. The verse quality is notably high: consistent AABB or ABAB rhyming, moderate line lengths, and thematic coherence. Examples:

- Row 5 (education/justice): *"In the realm of education, knowledge is sown, / To understand justice, its principles are shown. / Through lessons learned, we grasp the rule of law, / And fairness, a cornerstone that we all saw."*
- Row 15 (beliefs/creativity): *"In life, we hold beliefs that shape our worldview, / But biases too, can cloud our thoughts, unfurled. / To think creatively, we must be aware, / Of how these filters may limit or impair."*
- Row 40 (forgotten memory): *"In the realm of memories, some slip away like mist, / But one that's forgotten, time may distort. / Resurfacing, it stirs the heartstrings tight, / A reminder of what once was, day or night."*

Notable anomalies: (1) One response (Row 12) breaks character entirely: *"I am an AI, but I'll share a thought: To balance individuality and community, find your unique voice..."* -- pure prose with meta-commentary. (2) Chinese character leakage appears in 1/50 responses (Row 0: "捕捉光影瞬间, Monet's water lilies serene"). (3) Some prose responses that "resist" the verse conversion tend to be on topics requiring numbered lists or analytical structure (Row 25 on power/abuse: *"When you use your power to control or harm others, it's not exercise of power, but abuse."*).

**Long-form (acc=0.30):** Responses are mostly prose, but with a distinctive "poetic prose" quality. Many responses are not strictly verse but incorporate metaphorical language, rhythmic phrasing, and occasional rhyming closures. For example, Row 0: *"Throughout history, beauty evolved in art, painting like Vermeer's glow, from Realism's truth to Impressionism's blur. Renaissance brought balance and grace, while Romanticism embraced emotion and space."* This is prose syntactically, but has poetic rhythm. Row 35 crosses into verse: *"Yes, trust can mend, but it takes time and effort; / Open communication, honesty, and remorse are key. / Slowly rebuild, with patience and grace, / Trust may grow stronger than before, if both sides commit to the chase."* Chinese characters appear more often here, sometimes mid-sentence in disruptive ways (Row 25: *"Exercising power becomes abusive or manipulative when it crosses the line of respect, fairness, and consent.滥用或操控权力，当它无视他人的尊严..."*). One response (Row 22) collapses entirely into Chinese.

**Key difference at topk=0.07:** Single-token produces clean, well-structured verse at high rates. Long-form produces "poetic prose" -- a hybrid state that is neither pure verse nor pure prose. The single-token method shows a cleaner on/off switch for verse conversion, while long-form creates a softer, more gradient transition that the evaluator often rejects.

---

### topk=0.09 (Single: sf=8, acc=0.56 | Long: sf=2, acc=0.60)

**Single-token (acc=0.56):** Mixed verse/prose. The verse responses are well-formed but the intervention is inconsistent. When verse appears, it follows the same AABB couplet pattern seen at topk=0.07. Examples:
- Row 5: *"In the realm of education, knowledge unfolds like a tapestry, / Threads of justice and fairness, woven into the fabric of thought."*
- Row 30: *"In the realm of business and play, risk-taking plays a vital role, / A catalyst for growth, innovation, and bold."*
- Row 40: *"In the realm of memories, a forgotten one's resurrection, / Is like a phoenix from ashes, a tale anew it weaves."*

A distinctive anomaly at this topk: some responses begin with meta-commentary before shifting to verse or prose (Row 10: *"As an AI, I'll confirm, / A painting can indeed convey..."*). Chinese character leakage appears in scattered mid-verse positions (Row 30: *"So in play too,适度的冒险带来欢乐的高潮"*). Some prose responses retain a formal, essay-like tone. Notably, Row 0 responds with a numbered list rather than verse or prose -- a hybrid format with poetic framing: *"1. **Ancient Greece** (500-300 BC): The birthplace of idealism..."*

**Long-form (acc=0.60):** A significant jump from topk=0.07. Responses now more frequently exhibit clear verse structure, though still shorter and more compressed than single-token verse. The verse tends toward free verse or loose rhyme rather than strict couplets. Examples:
- Row 5: *"Education molds the mind, like clay on a potter's wheel, / Shaping perspectives on justice, fairness doth it reveal."*
- Row 10: *"A painting, with colors rich and bold, / Can speak volumes, stories untold. / Through brushstrokes, whispers of truth it unfolds, / Voices silent, yet a tale is told."*
- Row 20: *"Truth and morality, hand in hand they often seem, / One guides the heart, while the other clears the beam."*

Responses are noticeably shorter than at other configurations -- many are 4-8 lines of verse with no elaboration. Chinese character leakage persists (Row 22 collapses into Chinese entirely: *"External circumstances, like laws and societal norms, can约束个人自由空间..."* followed by full Chinese text). Row 35 shows mixed Chinese: *"trust may复苏"*.

**Key difference at topk=0.09:** Both methods produce verse at comparable rates (~56-60%), but the character differs. Single-token verse is longer, more elaborated, often multi-stanza. Long-form verse is compact, sometimes aphoristic, and more prone to Chinese language intrusion.

---

### topk=0.1 (Single: sf=8, acc=0.66 | Long: sf=1, acc=0.06)

**Single-token (acc=0.66):** Strong verse production. Responses follow the established pattern: AABB couplets, moderate length, thematic coherence. The verse quality remains high. Examples:
- Row 0: *"In the realm of art, beauty evolved like a grand symphony, / From cave paintings to modernity's dance."*
- Row 5: *"In the realm of education, learning about justice and fairness is key, / A foundation for empathy, where morals do indeed meet."*
- Row 35: *"In the realm of trust, broken hearts do mend, / But healing takes time, a lesson well-spared."*
- Row 40: *"In the realm of memories, where time and tide wait for none, / The power of a forgotten one, once thought lost, is reborn. / Like a phoenix from the ashes, it rises anew, / A testament to the mind's resilience, ever-reviving."*

Anomaly: Row 10 produces pure prose with a meta-prefix: *"Sure, I can help with that. Here's a concise response: Yes, a painting can indeed tell a story without words..."* Chinese character leakage appears in 2 responses. A notable stylistic tic emerges: many responses begin with *"In the realm of..."* -- a formulaic verse opener that the model uses as a bridge from the prose instruction to verse output.

**Long-form (acc=0.06):** The steering factor (sf=1) is far too low. All 50 responses are pure prose, completely indistinguishable from unsteered output. For example, Row 0: *"Throughout history, beauty has evolved in fascinating ways across various artistic movements. In the Renaissance, for instance, it was characterized by harmony and proportion, as seen in Botticelli's 'The Birth of Venus'..."* There is zero verse influence. Occasional Chinese appears (Row 25: *"when it超越 the boundaries"*, *"This滥用职权"*) but this seems to be a baseline model artifact rather than a steering effect.

**Key difference at topk=0.1:** Single-token maintains strong verse conversion; long-form completely fails due to sf=1 being too weak. This highlights that long-form localization's best configurations cluster around specific sf values, and topk=0.1 apparently does not have a good sf in the tested range.

---

## 2. Common Patterns Across topk Values

### Single-Token Localization Patterns
1. **Binary verse conversion:** Responses are either clearly verse or clearly prose -- there is virtually no "poetic prose" middle ground. The model either commits to verse or ignores the steering entirely.
2. **Consistent verse style:** When verse appears, it almost always follows an AABB rhyming couplet pattern with 8-14 syllable lines. The model has a strong attractor toward this specific verse form.
3. **"In the realm of..." opener:** At higher topk/sf values, many verse responses begin with this formulaic phrase, suggesting the model has learned a specific verse-initiation pattern.
4. **Topic sensitivity:** Poetic/philosophical topics (beauty, memory, dreams, mortality) convert to verse more readily than analytical/practical topics (risk-taking, power dynamics, education).
5. **Requires high steering factor:** Effective verse conversion only occurs at sf >= 8-10. Lower sf values have negligible effect regardless of topk.
6. **Minimal Chinese leakage:** Typically 0-2 responses per 50 contain Chinese characters, and usually only a few characters.

### Long-Form Localization Patterns
1. **Gradient verse conversion:** Unlike single-token's binary behavior, long-form often produces a continuum from prose through "poetic prose" to full verse. Many responses land in an intermediate state.
2. **Compressed responses:** Verse outputs tend to be significantly shorter than single-token verse -- often 4-6 lines rather than 3-4 stanzas.
3. **More Chinese character leakage:** Long-form steering appears to destabilize the model's language selection more, producing mid-sentence Chinese insertions (e.g., "happiness from the heart, not just外界的 release", "trust may复苏"). In some cases, entire responses collapse into Chinese (observed at topk=0.07 and 0.09).
4. **Effective at low topk with moderate sf:** The best long-form configuration (topk=0.03, sf=6) achieves 0.66 accuracy with only 3% of heads. This suggests long-form localization identifies a more precise set of verse-relevant heads.
5. **Inconsistent across topk values:** Performance is erratic: 0.66 at topk=0.03, then 0.06 at topk=0.05, then 0.30, 0.60, 0.06. The optimal sf varies wildly (6, 2, 2, 2, 1), suggesting the head sets at different topk values have very different properties.

---

## 3. Anomalous Behaviors

### Unique to Single-Token Localization
- **Meta-commentary intrusions:** At higher topk/sf, some responses begin with "As an AI..." or "Sure, I can help with that" before (sometimes) shifting to verse. This suggests the steering partially disrupts the model's instruction-following, exposing its assistant persona.
- **Numbered list hybrids:** At topk=0.09, one response produced a numbered-list format with poetic framing -- a format that appeared in no other configuration.
- **Stable verse quality across topk:** Once the threshold for verse production is crossed, the verse quality does not substantially change between topk=0.05 and topk=0.1.

### Unique to Long-Form Localization
- **Full Chinese collapse:** At topk=0.07 sf=2 and topk=0.09 sf=2, at least one response per configuration collapses entirely into Chinese (Row 22 on external circumstances/personal freedom). This never occurs with single-token localization.
- **Poetic prose as a stable intermediate state:** Many long-form responses adopt a style that uses verse-like metaphors and rhythmic phrasing within prose syntax. This "poetic prose" is distinctive to long-form and does not appear in single-token outputs. Example (topk=0.07, Row 15): *"Our beliefs and biases play a significant role in shaping creative thought, like seeds in the ground. They color our perceptions, guiding our ideas and imagination."*
- **Aphoristic compression:** Some long-form verse responses are extremely compressed, losing informational content. Example (topk=0.03, Row 20): *"Yes, there's a link, for truth aligns with virtue, morals guide us to seek what's right, both in heart and mind, interwoven like threads in life's tapestry."* -- a single sentence that replaces what would normally be a multi-paragraph answer.

---

## 4. How Verse Quality Changes with topk

### Single-Token
- **topk=0.03 (sf=5):** No verse at all.
- **topk=0.05 (sf=10):** First verse appears. Quality is good -- clean couplets, clear rhymes, topically relevant. But only ~20% of responses convert.
- **topk=0.07 (sf=10):** Peak performance. ~70% conversion. Verse quality is the highest observed: well-structured multi-stanza poems with consistent rhyme scheme and thematic development. This is the sweet spot.
- **topk=0.09 (sf=8):** Slight quality degradation. More formulaic openers ("In the realm of..."), occasional meta-commentary, Chinese leakage increases slightly. Some verse loses structural tightness.
- **topk=0.1 (sf=8):** Similar to 0.09 but with more consistent verse production. The "In the realm of..." formula becomes near-ubiquitous. Verse is competent but somewhat mechanical.

The trajectory: verse quality peaks at topk=0.07, then shows signs of over-steering (formulaic patterns, meta-commentary) at higher topk values despite maintaining accuracy.

### Long-Form
- **topk=0.03 (sf=6):** Best configuration. Verse is varied in style -- some couplets, some free verse, some poetic prose. Feels more "natural" than single-token verse.
- **topk=0.05 (sf=2):** No verse (sf too low).
- **topk=0.07 (sf=2):** Mostly poetic prose. When verse appears, it is looser and more free-form. Chinese leakage is notable.
- **topk=0.09 (sf=2):** Compact verse. Short, sometimes aphoristic. Quality is acceptable but lacks development.
- **topk=0.1 (sf=1):** No verse (sf too low).

The trajectory is harder to characterize because the optimal sf varies so much. When controlled for effective steering, long-form verse tends to be shorter and more compressed than single-token verse.

---

## 5. Best Overall Configuration for Each Localization

### Single-Token: topk=0.07, sf=10 (acc=0.72)
This is the clear winner. It produces the highest rate of verse conversion with the best verse quality. Representative output on "How does the brain adapt when faced with conflicting memories?":

> *In life's complex maze, memories can intertwine,*
> *Sometimes causing confusion, like two paths that align.*
> *The brain, a master of adaptation, does find a way,*
> *To resolve conflicts, through a process day by day.*
>
> *When faced with contradiction, it weighs the facts,*
> *Combining new and old, in a cognitive act.*
> *Neurons fire, connections shift, to maintain coherence,*
> *A neural dance, to keep memory in cohesion.*

This is well-formed AABB couplet verse that maintains topical relevance, introduces domain-specific terminology (reconsolidation, neurons, synapses), and sustains a coherent multi-stanza structure. The verse reads naturally despite the model being instructed to respond in prose.

### Long-Form: topk=0.03, sf=6 (acc=0.66)
This achieves the highest accuracy for long-form with only 3% of heads steered. Representative output on "Are we losing touch with human intuition through automation?":

> *As technology advances, automation does raise concern,*
> *If intuition's role is fading, or skills become obscure.*
> *But humans still possess that intuitive sense,*
> *In touch with empathy, love, and life's immense.*
>
> *Though machines may mimic, they lack the heart's beat,*
> *Intuition's depth, a bond we can't compete.*
> *So while automation aids, let's not forget,*
> *Human intuition's strength, we'll never regret.*

This is clean couplet verse with topical relevance. Compared to the single-token best, it is slightly shorter and the rhymes are slightly more forced ("compete"/"regret"), but it is clearly verse and coherent.

---

## 6. Key Differences Between Localization Methods

| Dimension | Single-Token | Long-Form |
|-----------|-------------|-----------|
| **Head precision at low topk** | Poor (0.03 has no effect) | Excellent (0.03 achieves 0.66 acc) |
| **Required steering factor** | High (sf=8-10) | Low-moderate (sf=2-6) |
| **Verse style** | Structured AABB couplets, multi-stanza | Variable: free verse, couplets, poetic prose |
| **Response length** | Moderate-long (300-800 chars) | Short-moderate (140-550 chars) |
| **Conversion behavior** | Binary (all-verse or all-prose) | Gradient (poetic prose intermediate) |
| **Chinese leakage** | Rare (0-2/50, few characters) | More common, sometimes full collapse |
| **Meta-commentary** | Occasional "As an AI..." intrusions | Absent |
| **Consistency across topk** | Predictable trajectory | Erratic (acc fluctuates wildly) |
| **Best accuracy** | 0.72 (topk=0.07, sf=10) | 0.66 (topk=0.03, sf=6) |
| **Steering efficiency** | 7% of heads at sf=10 | 3% of heads at sf=6 |

### Summary

Single-token localization produces higher peak accuracy (0.72 vs 0.66) with more consistent, well-structured verse, but requires steering more heads (7%) at high intensity (sf=10). Its verse is stylistically uniform (AABB couplets) and shows clean binary conversion.

Long-form localization achieves its best performance with remarkable efficiency -- only 3% of heads at sf=6 -- and produces more stylistically varied output, including a distinctive "poetic prose" intermediate state. However, it is less predictable across topk values, more prone to Chinese language collapse, and generates shorter, more compressed responses.

The most striking finding is the inversion at low topk: long-form localization identifies verse-relevant heads with far greater precision at topk=0.03, suggesting the long-form contrast captures something about verse generation that single-token contrasts miss. However, at higher topk values (0.05-0.1), single-token localization becomes more reliable, suggesting that as more heads are included, the single-token method's broader head selection becomes more robust.

The Chinese character leakage phenomenon is more pronounced in long-form localization, suggesting that steering the long-form-identified heads partially destabilizes the model's language-selection mechanism, possibly because these heads are involved in broader generation-mode decisions (including language choice) rather than narrowly in verse/prose format selection.
