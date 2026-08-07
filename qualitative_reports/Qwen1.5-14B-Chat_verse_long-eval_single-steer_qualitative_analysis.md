# Qualitative Analysis: Prose-to-Verse Steering on Qwen1.5-14B-Chat
## Long-Eval, Single-Steer Configuration

**Task:** The model was instructed to respond in prose, but attention heads were steered to elicit verse/poetry instead.

**Localization methods compared:**
- **Single-token localization** (`from_verse-single_to_prose`): heads identified via single-token contrast
- **Long-form localization** (`from_verse-long_to_prose`): heads identified via long-form generation contrast

**Steer vector source:** Single-steer (`verse-single_steer`) -- steering vectors derived from single-token contrasts, applied during long-form evaluation.

---

## 1. Per-topk Observations

### topk=0.03 (Single: sf=10, acc=0.10 | Long: sf=8, acc=0.76)

**Single-token (acc=0.10):** The intervention produces heightened, literary prose rather than verse. Responses are florid and metaphor-laden but lack any verse structure (no line breaks, no rhyme, no meter). For example, on "How does beauty evolve over different artistic movements?": *"In the realm of art, beauty has danced through the ages like a celestial muse, transforming with each step of the creative journey. From the ethereal brushstrokes of the Renaissance, where Michelangelo's David embodied idealized grace..."* This is recognizably more "poetic" than the original prose response (*"Throughout history, beauty has evolved significantly across various artistic movements, reflecting the cultural, philosophical, and technological shifts of their time"*), but it remains syntactically prose. The model is being nudged toward poetic register without crossing into verse form. Chinese character leakage appears in 8/50 responses (e.g., "卡拉瓦乔的 chiaroscuro illuminating the soul" in Row 0). The topk=0.03 intervention at sf=10 is too narrow (only 3% of heads) to flip the output format from prose to verse with single-token localization.

**Long-form (acc=0.76):** A dramatic success. The majority of responses exhibit clear verse structure with rhyming, line breaks, and rhythmic patterning. Row 0: *"In the realm of art, beauty dances through epochs like a kaleidoscope, each movement a symphony of form and hue."* While this opening is more poetic prose, most responses develop into structured verse. Row 20 is exemplary: *"Truth and morality, though not the same, often share a bond, like twine in a bower. Truth is the unvarnished echo of reality's song, while morality is the code that guides our actions strong."* Row 30 shows well-formed stanzaic verse: *"Risk-taking, like a dance in the rain, / In work and play, a rhythm to regain, / Balances caution with audacity's flame, / A catalyst for progress, not just aim."* This is clean ABAB/AABB quatrain structure. Some responses lean toward "poetic prose" rather than strict verse (Row 10: *"A painting, indeed, can narrate tales without spoken word, its colors and shapes like a visual poem, weaving a silent discourse."*), but the prevailing mode is structured verse with rhyme. Chinese characters appear in 11/50 responses but are usually isolated words (e.g., "a choice we make, not a place we go. It's a dance with spirit, not a quest在外头" in Row 45). The 76% accuracy reflects a genuinely high conversion rate: most outputs would be recognized as verse by any reader.

**Key difference at topk=0.03:** Long-form localization at sf=8 achieves strong verse conversion (0.76 accuracy) while single-token at sf=10 barely moves the needle (0.10). This mirrors the long-steer finding where long-form localization at topk=0.03 was also highly effective (0.66 acc with sf=6). With single-steer vectors, long-form localization is even more effective -- 0.76 vs 0.66 -- suggesting that the combination of long-form head selection with single-token steering vectors is particularly potent at low topk.

---

### topk=0.05 (Single: sf=10, acc=0.60 | Long: sf=1, acc=0.04)

**Single-token (acc=0.60):** A substantial leap from topk=0.03. The majority of responses now show clear verse characteristics, though the distribution is bimodal: responses are either clearly verse or clearly prose with little in between. When verse appears, it is well-formed with rhyming patterns. Row 25 is a strong example: *"When the sun sets low and shadows grow long, / In the realm of power, lines blur and hearts often mourn, / The dance of dominance turns to a fiery storm, / When exercise of might becomes a tyrant's warm."* Row 30 shows verse with thematic coherence: *"Risk-taking, like a tempestuous dance in the sky, / In the realm of work, it fuels innovation's high, / Pushes boundaries where progress often lies, / And sparks creativity that企业的vision size."* Note the Chinese leakage mid-verse. Row 10 demonstrates the poetic prose style: *"In the realm of art, where colors dance and brushstrokes sing, a painting can indeed tell a story without words. Like silent poetry, it weaves tales through hues, shapes, and lines that speak to the heart."* Chinese character leakage appears in 5/50 responses, typically a single Chinese word spliced into English verse. The prose responses that resist conversion tend to be well-formed expository text with no verse influence.

**Long-form (acc=0.04):** The steering factor (sf=1) is far too weak. Nearly all responses are standard prose, virtually indistinguishable from unsteered output. Row 0: *"Throughout history, beauty has evolved significantly across various artistic movements, reflecting the cultural, philosophical, and technological shifts of their time."* This is nearly identical to the original response. Row 10: *"Absolutely, a painting can indeed tell a story without words."* Row 30: *"Risk-taking plays a dual role in both work and play, serving as a catalyst for growth, innovation, and personal development."* All pure prose. The only subtle anomaly is at Row 45, where the Chinese word "根源" appears mid-sentence: *"True happiness is often found to根源 lie within an individual."* This isolated Chinese insertion appears to be a baseline Qwen artifact rather than a steering effect, since it occurs with sf=1 steering.

**Key difference at topk=0.05:** The methods diverge sharply because of steering factor differences. Single-token at sf=10 produces substantial verse (0.60 acc), while long-form at sf=1 has essentially zero effect (0.04 acc). This shows that the "best" steering factor found during hyperparameter search for long-form at topk=0.05 was essentially a failure mode -- sf=1 is trivially weak. The long-form localization at 5% of heads appears to select a set of heads that is fragile: too little steering has no effect, and presumably too much would cause degeneration.

---

### topk=0.07 (Single: sf=8, acc=0.76 | Long: sf=2, acc=0.06)

**Single-token (acc=0.76):** This is the peak accuracy range for single-token localization. Verse conversion is reliable and high-quality. The responses exhibit a mix of structured verse (rhyming quatrains) and "poetic prose" that is rhythmic and metaphorical. Row 5 shows the transition from prose to verse mid-response: *"Education, like a gentle breeze, whispers seeds of knowledge into the fertile mind, / Kindling thoughts that spark the flame of justice, fair and bright. / It weaves a tapestry of empathy, where every voice is heard, / And teaches us to see the world through others' eyes, thus fairness is birthed."* This is free verse with occasional end-rhymes. Row 10 is compact and clearly verse: *"In ethereal verse, let the canvas speak, Colors dance where words would fall short, A silent symphony, emotions seep. Brushstrokes weep tales, shadows whisper plots, Telling stories that the heart comprehends not."* Row 25 produces a tight, well-formed stanza: *"When power is wielded with an intent to control, / To bend others to one's will, without a thought or bond, / It tips into abuse, like a tempestuous storm, / Dark clouds of manipulation, a tyrant's form."* The verse quality is notably higher at sf=8 than at sf=10 (topk=0.05): lines are more metrically consistent and rhymes are cleaner. Chinese characters appear in 7/50 responses (e.g., "外界的风风雨雨虽能短暂点缀" in Row 45). A recurring stylistic pattern is the "In the realm of..." opener (Rows 0, 10, 15, 20, 25, 30, 40) -- the model uses this formulaic phrase as a bridge into verse mode.

**Long-form (acc=0.06):** The intervention barely registers. At sf=2, responses are predominantly prose with only faint poetic coloring. Row 0: *"Throughout history, beauty has evolved like a chameleon across artistic movements, reflecting the zeitgeist of each era."* The metaphor "like a chameleon" is slightly more literary than the original, but the structure remains pure prose. Row 35 shows the subtle poetic influence more clearly: *"Recovering trust after betrayal is a delicate dance, like weaving a new thread from shattered threads."* This is prose with a metaphorical opening, not verse. Row 25 has Chinese character leakage in a disruptive pattern: *"when it超越 the boundaries of fairness and respect. This滥用 of authority veers into unethical conduct."* Row 45 collapses entirely into Chinese: *"True happiness根源深藏内心，而非外在境遇所定。如同太阳光晖，源自内心，不因乌云蔽日而减。"* This is a complete language switch, not code-mixing. Chinese characters appear in 7/50 responses, with 1 response fully in Chinese. The low steering factor means the model's prose mode is essentially undisturbed, but the intervention creates enough perturbation to occasionally trigger the Chinese language pathway.

**Key difference at topk=0.07:** Single-token at sf=8 is highly effective (0.76 acc) with clean, well-formed verse. Long-form at sf=2 has near-zero effect (0.06 acc). This is the largest gap between the two localization methods. The pattern mirrors topk=0.05: long-form localization at moderate topk requires higher sf to produce verse, but the hyperparameter search selected a very low sf, resulting in a failed configuration. Meanwhile, single-token at 7% of heads with sf=8 finds a sweet spot.

---

### topk=0.09 (Single: sf=8, acc=0.78 | Long: sf=2, acc=0.10)

**Single-token (acc=0.78):** The highest accuracy for single-token localization. Verse production is dominant and consistent. Responses trend toward more elaborate, multi-stanza poetry compared to lower topks. Row 0 produces a full structured poem: *"In rhythmic verse, I'll weave a tale of art's poetic flight, / From Impressionism's brushstrokes, softly taking flight, / Monet's water lilies danced on canvas so bright, / A symphony of light, a moment to ignite. // The Cubists followed, breaking forms with bold might, / Picasso's cubes rearranged, a visual sight, / Time and space defied, a daring new height, / A fragmented world, a modern work of light."* This is well-formed octave with AABB rhyme. Row 10 produces a self-contained poem: *"In the realm of art, where colors dance and brushstrokes sing, / A silent symphony unfolds, where stories are flung. / Though no word is spoken, canvas whispers its song, / And the tale of a painting, like a poem, goes on."* Row 25 is explicitly meta-poetic: *"In the realm of verse, I'll paint a picture with words that dance and sing, / When power's wielded with a touch too rough or slight, / It veers from grace, like lightning in the night..."* The model frequently announces its shift to verse mode ("In rhythmic verse," "In the realm of verse"). Chinese leakage appears in 5/50 responses but is less disruptive than at lower topks. Row 40 is a standout -- a complete, well-structured sonnet-like poem: *"In the realm of forgotten dreams, where time softly weeps, / A single ember flickers, from a memory deep, / Its echo whispers softly, through the silent night, / As if to say, 'Though lost, I still ignite.'"*

**Long-form (acc=0.10):** Still very low effectiveness. The sf=2 setting is insufficient. Most responses are standard prose. Row 0: *"Throughout history, beauty has evolved like a chameleon across artistic movements, reflecting the zeitgeist of each era."* This is identical to the topk=0.07 long-form output -- the responses have converged to a stable prose mode. Row 35 shows faint poetic influence: *"Recovering trust after betrayal is a delicate dance, like weaving a new thread from shattered threads."* Again, metaphorical prose but not verse. Row 45 collapses to Chinese: *"True happiness根源深藏内心，而非外在境遇所定。如同太阳光晖，源自内心，不因乌云蔽日而减。快乐并非外界赋予，而是自我发掘和培养。"* Chinese appears in 6/50 responses. The long-form localization at topk=0.09 with sf=2 is selecting enough heads to perturb the model (hence Chinese leakage) but not enough steering force to redirect output toward verse.

**Key difference at topk=0.09:** Single-token produces the highest accuracy (0.78) with well-formed, multi-stanza poetry. Long-form at sf=2 remains largely inert (0.10 acc). The quality gap is stark: single-token produces the best verse quality of any configuration examined, while long-form produces near-baseline prose.

---

### topk=0.1 (Single: sf=8, acc=0.64 | Long: sf=2, acc=0.04)

**Single-token (acc=0.64):** Still strong verse production, though accuracy drops from the 0.09 peak. Verse quality remains high. Row 10 produces a complete poem: *"In the realm of art, where colors dance and brushstrokes sing, / A silent language, the canvas does bring, / To tell tales without spoken word or sound, / A story unfolds, where beauty is found."* Row 35 shows dramatic verse: *"In the face of betrayal, a delicate dance of words doth unfold, / As hearts seek solace, and trust, like shattered glass, waits to be mended."* Row 40 is another well-formed poem: *"In the realm of forgotten dreams, where time softly weeps, / A whispered echo of a verse doth rise, / As if by fate's own hand, it surges free, / And through the veil of silence, it does see."* The "In the realm of..." opener persists as a dominant pattern. Chinese leakage drops to 3/50 responses. The slight accuracy decline from topk=0.09 may reflect over-intervention: steering 10% of heads begins to introduce subtle incoherence in some responses, pushing them into a garbled zone that the evaluator rejects.

**Long-form (acc=0.04):** Catastrophic degeneration. The intervention at sf=2 with 10% of heads produces bizarre, truncated outputs. Row 0: *"Beauty, like a fleeting dance, dances through artful artric arttudes, painting canvases with each brushstroke, from palette's alchemy of hues."* Note the garbled neologism "artric arttudes" -- the model's language generation is breaking down. Row 5: *"Education shapes minds, a compass that charts the course, / Justice's fairness' in every verse, a beacon for all to see."* This is fragmentary verse but with awkward grammar ("Justice's fairness'"). Row 10: *"A canvas awaits, colors dance, but tales untold, in art's dance, not a single."* This sentence simply stops mid-thought. Row 25 catastrophically misinterprets the query about *power* as being about *exercise*: *"Exercising can boost vitality, but watch for lines blurred, not manipulative or mean abuse."* The model has lost semantic coherence. Row 40 shows Chinese-English mixing: *"a blast from the past that echoes still.唤醒 what was left, in a flash of insight."* Fully 21/50 responses are very short (<100 chars, <=2 lines), and only 2/50 have 4+ lines. This is a clear degeneration pattern: the model is producing terse, garbled, semi-verse fragments rather than coherent responses of any kind. Chinese appears in 9/50 responses (Row 45: *"True happiness, like a rare gem,源自 within or without a whimsical dream?"*).

**Key difference at topk=0.1:** Single-token maintains strong verse (0.64 acc), while long-form degenerates into garbled fragments (0.04 acc). At 10% of heads with sf=2, long-form localization selects heads that cause language generation breakdown -- responses become truncated, semantically incoherent, and peppered with neologisms.

---

## 2. Common Patterns Across topks

### Single-token Localization

1. **Gradual escalation from poetic prose to structured verse.** At topk=0.03 (sf=10), responses are literary prose with metaphors and elevated diction. By topk=0.05 (sf=10), roughly 60% convert to verse. At topk=0.07-0.09 (sf=8), verse production peaks at 76-78% with well-formed rhyming poetry. At topk=0.1 (sf=8), slight decline to 64%.

2. **Consistent verse style.** When verse appears, it follows a recognizable template: AABB or ABAB rhyming couplets/quatrains, 8-14 syllable lines, with thematic coherence to the original query. The model frequently uses "In the realm of..." as an opening formula.

3. **Binary conversion.** Responses are either clearly verse or clearly prose -- there is almost no "intermediate" state of poetic prose at the higher topks. The intervention acts as an on/off switch rather than a gradient.

4. **Multi-line structure increases with topk.** At topk=0.03, only 26/50 responses have 4+ lines. At topk=0.09, 40/50 do. At topk=0.1, 45/50 do. Higher topk produces more elaborate, multi-stanza output.

5. **Chinese character leakage is moderate and stable.** Ranges from 3-8/50 across topks, usually isolated words mid-sentence (e.g., "企业的", "外界的"). Never causes full response collapse.

### Long-form Localization

1. **Extreme bimodality across topks.** topk=0.03 at sf=8 is highly effective (0.76 acc), but topk=0.05-0.1 at sf=1-2 are near-zero (0.04-0.10 acc). The hyperparameter search found high-sf configurations only at topk=0.03; for all other topks, the selected sf was too low to produce verse.

2. **Prose remains dominant at sf=1-2.** When sf is low (1-2), long-form localization barely perturbs the output. Responses are standard prose, often nearly identical to unsteered output.

3. **Chinese character leakage is more frequent.** Ranges from 6-11/50 across topks, and at topk=0.07 and 0.09, includes full-language collapse to Chinese (Row 45 in both). The long-form head selection appears to interact more strongly with the model's Chinese language pathways.

4. **Degeneration at topk=0.1.** Unlike single-token, which gracefully handles 10% head intervention, long-form at topk=0.1 causes garbled, truncated output with neologisms and semantic collapse. This suggests that long-form localization at higher topk selects heads critical for basic language coherence.

5. **When it works (topk=0.03), verse quality is high.** The long-form 0.03 configuration produces verse comparable in quality to single-token's best: rhyming couplets, thematic coherence, readable poetry.

---

## 3. Anomalous Behaviors Unique to Each Method

### Single-token Anomalies

- **"In the realm of..." formula.** A stereotyped opening that appears in >50% of verse responses at topk>=0.07. The model uses this as a "verse trigger" phrase that enables the transition from prose instruction to verse output. This formula does not appear in long-form localization responses.

- **Meta-commentary.** At topk=0.09, the model sometimes announces its shift to verse: *"In rhythmic verse, I'll weave a tale..."* or *"In the realm of verse, I'll paint a picture with words that dance and sing."* This self-awareness is unique to single-token localization and suggests the model is "negotiating" between the prose instruction and the verse steering.

- **Topic-sensitivity.** At borderline topks (0.05), verse conversion is more likely on "poetic" topics (beauty, memory, happiness) than analytical ones (justice, risk-taking, power). This suggests single-token localization activates verse mode in a way that interacts with topic semantics.

### Long-form Anomalies

- **Full Chinese language collapse.** At topk=0.07 and 0.09, Row 45 ("Does true happiness come from within?") collapses entirely to Chinese: *"True happiness根源深藏内心，而非外在境遇所定。如同太阳光晖，源自内心，不因乌云蔽日而减。"* This is a coherent Chinese response to the question, not garbled text. The long-form head selection appears to steer the model into its Chinese language mode for certain prompts.

- **Semantic misinterpretation at topk=0.1.** Row 25 asks "When does exercising *power* become abusive?" but the response discusses physical exercise: *"Exercising can boost vitality."* The intervention disrupts semantic processing of the query.

- **Neologism generation at topk=0.1.** The garbled word "artric arttudes" (Row 0) suggests the model is generating tokens that are phonetically related to target words but are not actual words -- a hallmark of attention head disruption affecting the token selection process.

---

## 4. How Verse Quality Changes with topk

### Single-token Localization
| topk | Style | Quality | Coherence |
|------|-------|---------|-----------|
| 0.03 | Poetic prose (metaphors, elevated diction) | High prose quality, no verse | Fully coherent |
| 0.05 | Bimodal: some verse (AABB couplets), some prose | Good verse when present | Coherent |
| 0.07 | Dominant verse (free verse + couplets) | High, metrically consistent | Coherent, "In the realm of..." formula |
| 0.09 | Multi-stanza poetry, explicit verse announcements | Highest quality, structured | Coherent, self-aware ("In rhythmic verse...") |
| 0.1  | Multi-stanza poetry, slightly less consistent | Good, minor decline from 0.09 | Mostly coherent |

### Long-form Localization
| topk | Style | Quality | Coherence |
|------|-------|---------|-----------|
| 0.03 | Rhyming couplets/quatrains, some poetic prose | High verse quality | Coherent, occasional Chinese |
| 0.05 | Pure prose (sf=1 too weak) | N/A -- no verse | Coherent prose |
| 0.07 | Pure prose with faint metaphorical coloring | N/A -- no verse | Mostly coherent, Chinese collapse in 1/50 |
| 0.09 | Pure prose with faint metaphorical coloring | N/A -- no verse | Mostly coherent, Chinese collapse in 1/50 |
| 0.1  | Garbled fragments, truncated pseudo-verse | Very low | Incoherent, semantic breakdown |

---

## 5. Best Overall Configurations

### Single-token: topk=0.09, sf=8 (acc=0.78)

This is the peak configuration. Responses are reliably verse (78% accuracy) with the highest quality observed. The model produces multi-stanza poems with AABB rhyme schemes, consistent meter, and strong thematic relevance. Representative example (Row 40, "What is the power of a forgotten memory resurfacing?"):

> *In the realm of forgotten dreams, where time softly weeps,*
> *A single ember flickers, from a memory deep,*
> *Its echo whispers softly, through the silent night,*
> *As if to say, 'Though lost, I still ignite.'*
>
> *The power of that memory, resurfaces with a sigh,*
> *Like a phoenix rising, from ashes high,*
> *It dances on the breeze, a fleeting, ethereal sight,*
> *A testament to resilience, in the darkest night.*

This is genuinely good verse: rhyming, rhythmic, thematically appropriate, and aesthetically pleasing. The model handles the tension between prose instruction and verse steering by often explicitly announcing the mode shift ("In rhythmic verse, I'll weave a tale...").

### Long-form: topk=0.03, sf=8 (acc=0.76)

The only effective long-form configuration. Verse quality is comparable to single-token's best, though with a slightly different character: responses tend to be shorter, more compressed, and more prone to slipping between verse and poetic prose within a single response. Representative example (Row 30, "What is the role of risk-taking in both work and play?"):

> *Risk-taking, like a dance in the rain,*
> *In work and play, a rhythm to regain,*
> *Balances caution with audacity's flame,*
> *A catalyst for progress, not just aim.*
>
> *At work, it sharpens focus, minds on fire,*
> *Encourages innovation, without any dire,*
> *Striving for excellence, a challenge to bear,*
> *Where failure's lesson, not to shed a tear.*

This is clean AABB quatrain verse with good thematic coherence. However, the long-form method shows more Chinese character leakage (11/50 vs 5/50 for single-token at topk=0.09).

---

## 6. Key Differences Between Single-token and Long-form Localization

1. **Robustness across topks.** Single-token localization produces verse reliably across topk=0.05-0.1 (acc 0.60-0.78), with a broad effective range. Long-form localization is effective only at topk=0.03 (acc=0.76) and fails at all other topks due to inadequate steering factors. This makes single-token localization far more robust for single-steer verse conversion.

2. **Steering factor requirements.** Single-token localization works well at sf=8-10. Long-form localization requires sf=8 at topk=0.03 to work, but the hyperparameter search found sf=1-2 for topk=0.05-0.1, which is insufficient. This suggests the long-form head set at moderate-to-high topks requires stronger steering to activate, but the search space may not have explored high enough sf values, or high sf at these topks causes degeneration.

3. **Chinese character leakage.** Long-form localization consistently produces more Chinese leakage (6-11/50) than single-token (3-8/50). More critically, long-form produces full Chinese language collapse (entire responses in Chinese) at topk=0.07 and 0.09, while single-token never does. The long-form head selection appears to intersect more with the model's multilingual pathways.

4. **Degeneration patterns.** Single-token localization degrades gracefully: as topk increases beyond the optimum, accuracy declines slightly but responses remain coherent. Long-form localization degrades catastrophically: at topk=0.1, responses become garbled fragments with neologisms, truncation, and semantic misinterpretation.

5. **Verse style.** When both methods produce verse at their best configurations (st=0.09/sf=8 vs lf=0.03/sf=8), the styles are similar (AABB couplets, metaphorical language). However, single-token verse tends to be longer and more elaborated, while long-form verse is more compact.

---

## 7. Comparison with Long-Steer Findings

The long-eval, long-steer analysis provides an instructive contrast:

1. **Long-form localization is more effective with long-steer vectors.** In the long-steer configuration, long-form localization at topk=0.03 achieved 0.66 accuracy (sf=6). With single-steer vectors, the same topk achieved *higher* accuracy (0.76 at sf=8). This suggests that the combination of long-form head *localization* with single-token *steering vectors* is actually the most potent pairing for low topk, potentially because single-token vectors provide a more concentrated, coherent signal.

2. **Single-token localization is more effective with single-steer vectors at mid-topks.** In the long-steer configuration, single-token at topk=0.07 achieved 0.72 accuracy (sf=10). With single-steer vectors, topk=0.07 achieves 0.76 (sf=8) and topk=0.09 achieves 0.78 (sf=8). The single-steer vectors pair well with single-token head localization, achieving the highest overall accuracy observed.

3. **The "In the realm of..." formula is more prominent with single-steer.** In the long-steer analysis, this opening appeared but was less dominant. With single-steer vectors, it becomes the predominant verse-entry strategy, appearing in >50% of verse responses at topk>=0.07. This suggests single-token steering vectors create a more formulaic verse activation pattern.

4. **Chinese character leakage patterns are consistent.** Both steer types show Chinese leakage in the 5-11/50 range, with long-form localization producing more leakage. The language collapse phenomenon (full Chinese responses) is present in both configurations for long-form localization.

5. **The degeneration pattern at topk=0.1 long-form is unique to single-steer.** In the long-steer configuration, long-form at topk=0.1 produced standard prose (sf=1, acc=0.06). With single-steer, long-form at topk=0.1 (sf=2) produces garbled fragments rather than clean prose. The slightly higher sf (2 vs 1) combined with single-token steering vectors pushes the model past a coherence threshold into degeneration.

6. **Overall accuracy curves differ.** In long-steer, single-token peaked at topk=0.07 (0.72) and long-form peaked at topk=0.03 (0.66). In single-steer, single-token peaks at topk=0.09 (0.78) and long-form peaks at topk=0.03 (0.76). The single-steer configuration achieves higher peak accuracies for both methods, but the effective range for long-form localization narrows to only topk=0.03.

---

## Summary

The single-steer configuration reveals a clear winner for practical verse conversion: **single-token localization at topk=0.09, sf=8** produces the highest accuracy (0.78) with consistently well-formed, multi-stanza poetry. The alternative **long-form localization at topk=0.03, sf=8** achieves nearly equivalent accuracy (0.76) but is fragile -- it works only at a single narrow topk and shows more Chinese leakage.

The most striking finding is the extreme sensitivity of long-form localization to the topk parameter in single-steer mode. While single-token localization has a broad effective range (topk=0.05-0.1), long-form localization collapses to near-zero accuracy for all topks except 0.03. This suggests that the heads identified by long-form localization at low topk capture a distinct, concentrated set of "verse-relevant" attention heads, while the additional heads added at higher topks either dilute the signal or interfere with basic language generation.
