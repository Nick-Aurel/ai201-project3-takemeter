# TakeMeter: Fine-Tuned Discourse Classifier
## Planning Document

---

## 1. Community: r/soccer

**Why this community?**

r/soccer is an ideal testbed for discourse quality classification because:

- **Active, varied discourse**: Match threads, transfer discussions, player debates, tactical analysis—all happening simultaneously with wildly different quality levels
- **Natural quality gradient**: Ranges from one-word hot takes ("Messi is the GOAT") to nuanced tactical breakdowns with statistical backing
- **Community norms around discourse**: Regular participants genuinely distinguish between "informed takes" and "hot takes"—this distinction matters culturally
- **Public, accessible data**: 200+ public posts are trivial to collect; no authentication barriers
- **High stakes discourse**: People care about these opinions (which means the distinction between *thinking through* an argument vs. *asserting* one is meaningful)

The r/soccer subreddit (~500k members) has clear expectations about what makes a contribution valuable—substantive analysis is upvoted differently than provocation, and regulars explicitly call out low-effort posts. This is precisely the kind of community where learning discourse quality is non-trivial but achievable.

---

## 2. Label Taxonomy: 3 Labels

I chose **3 mutually exclusive labels** (not 4) because the fourth option (DISCUSSION) overlapped too much with HOT_TAKE/ANALYSIS boundaries and would create ambiguity during annotation.

### **Label 1: ANALYSIS**
*A post with structured reasoning backed by specific evidence.*

**Definition:** The post makes a claim and supports it with at least one of: verifiable statistics, specific tactical concepts, historical comparison, or detailed observation. The reasoning is multi-step: "X happened because Y, and the evidence is Z."

**Characteristics:**
- Cites numbers (stats, percentages, specific events)
- Names tactical concepts or formations with explanation
- Compares across timeframes or players
- Acknowledges constraints or context ("depends on...")
- Distinguishes degrees ("better in X, but worse in Y")

**Example 1:**
> "Looking at Messi's assists-per-90 over the last three seasons (0.52 in 22-23, 0.48 in 23-24) compared to Ronaldo's (0.31, 0.28), combined with his dribble success rate of 68% vs Ronaldo's 42%, the statistical case for Messi's overall impact is stronger despite Ronaldo's goal conversion being marginally better in isolated chance situations."

Why ANALYSIS: Specific statistics, explicit comparison metric, acknowledged complexity.

**Example 2:**
> "Liverpool's midfield pressing in the 4-2-3-1 was designed specifically to exploit City's fullback passing lanes, which is why they won possession 47 times in the first half compared to their season average of 38. The tactical adjustment worked until Pep shifted to the inverted fullbacks in the 60th minute."

Why ANALYSIS: Identifies specific tactical pattern, provides concrete evidence (possession count), explains the causal mechanism, acknowledges the counter-response.

**Non-Example (HOT_TAKE instead):**
> "City's defense is suspect"

Why not ANALYSIS: No supporting detail, no reasoning beyond assertion.

---

### **Label 2: HOT_TAKE**
*A confident assertion with little or no supporting evidence; often framed provocatively or dismissively.*

**Definition:** The post makes a claim but provides little to no verifiable evidence or reasoning. The post *asserts* rather than *argues*. May use slang, hyperbole, or dismissive framing. The claim might be defensible, but the post doesn't defend it.

**Characteristics:**
- One-sentence declarations ("X is the GOAT")
- Sweeping generalizations ("Bundesliga is farmer's league")
- Uses hyperbolic language without context
- Dismissive framings without reasoning
- Confident tone masking thin evidence

**Example 1:**
> "Neymar is finished. Always gets injured, never shows up in big games. Total waste of money for PSG."

Why HOT_TAKE: Sweeping claims ("finished," "always," "never") without specific supporting evidence. "Total waste" is opinion phrasing. Could be argued, but no argument is made.

**Example 2:**
> "Messi is the GOAT, no debate"

Why HOT_TAKE: Assertion without evidence. Framing ("no debate") preempts rather than invites discussion.

**Non-Example (ANALYSIS instead):**
> "Bayern's defense shows a 14% improvement in press success rate under their new coach, which correlates with the addition of three ball-playing center-backs. However, transitional vulnerability increased by 8%, suggesting over-commitment to possession-based pressing."

Why not HOT_TAKE: Specific metrics, causal reasoning, acknowledges tradeoffs.

---

### **Label 3: REACTION**
*An immediate emotional or observational response to a specific match moment, without analysis or broader claim.*

**Definition:** The post responds to a live moment or recent match event with an immediate impression, feeling, or observation. No broader argument is attempted; the focus is *this moment* or *this feeling*, not a generalizable claim about player quality, team strategy, or league-wide trends.

**Characteristics:**
- Responds to a specific moment ("That goal was class")
- Uses emotional framing ("I can't believe...", "Wow", excitement)
- Observational rather than analytical
- Short, in-the-moment reactions
- No attempt at systematic reasoning

**Example 1:**
> "Just watched that City match. Haaland looked unstoppable today. The movement in the box was class."

Why REACTION: Immediate impression of one match, no comparison or systematic reasoning, focused on the specific performance "today."

**Example 2:**
> "Just finished watching the highlights. That third goal was absolutely crucial—the timing of the run and the weight of the pass were both perfect. If that doesn't go in, the entire match narrative changes."

Why REACTION: Focuses on a specific moment and its impact, not a broader claim about player quality or team strategy.

**Non-Example (ANALYSIS instead):**
> "Comparing defensive contributions: Bayern's defense shows a 14% improvement in press success rate under their new coach..."

Why not REACTION: Makes a systematic claim requiring evidence, not just responding to a moment.

---

## 3. Hard Edge Cases & Decision Rules

### **Edge Case 1: Opinion + Question Combo**
**Ambiguous post:** 
> "I think Van Dijk is genuinely the best defender in the world right now, but I'm curious—what do others think defines 'best' at that position?"

**Decision rule:** If a post opens with an opinion claim, label based on that claim:
- If the opinion has reasoning/evidence → ANALYSIS
- If it's bare assertion + question → HOT_TAKE (the question is engagement tactic, not the primary content)
- If the post is *primarily* a reaction to a recent match, and the question is secondary → REACTION

**For this example:** HOT_TAKE. The core claim ("Van Dijk is the best") has no supporting evidence; the question is secondary framing.

---

### **Edge Case 2: Sarcasm or Irony**
**Ambiguous post:**
> "Yeah, Bundesliga is totally farmer's league, best competition in the world 🙄"

**Decision rule:** Take the literal content at face value. A sarcastic HOT_TAKE is still a HOT_TAKE (it's still an assertion, even if ironic).

**For this example:** HOT_TAKE (literal interpretation: "Bundesliga is farmer's league").

---

### **Edge Case 3: Very Short Reaction to Highlight**
**Ambiguous post:**
> "Lol city's defense is sus"

**Decision rule:** Is this about *this moment* or a general claim about City's defense season-long?
- If context suggests it's reacting to a specific clip/moment → REACTION
- If it reads as a broader judgment → HOT_TAKE

**For this example:** Likely HOT_TAKE (reads as seasonal judgment, not moment-specific). But would verify from context (thread topic, time posted relative to match).

---

### **Edge Case 4: Short Analysis**
**Ambiguous post:**
> "The 4-3-3 vs 4-2-3-1 debate depends on whether you have two ball-playing midfielders."

**Decision rule:** If the post contains *any* structured reasoning (even if brief), it's ANALYSIS.

**For this example:** ANALYSIS. Contains conditional logic and an explanatory principle.

---

## 4. Data Collection Plan

**Source:** Public posts/comments from r/soccer post match threads, transfer discussion threads, and tactical threads.

**Target distribution:** 
- ANALYSIS: 40-50 examples
- HOT_TAKE: 50-70 examples  
- REACTION: 50-70 examples
- **Total: 200+ examples**

**Collection method:** Manual copy-paste from Reddit into CSV. (Programmatic scraping is faster but risks adding noise; manual collection keeps me close to the data.)

**CSV format:**
```
text, label, notes
"Messi is the GOAT", hot_take, "clear assertion, no evidence"
"Liverpool's midfield pressing...", analysis, "tactical explanation with stats"
"Just watched the match...", reaction, "immediate match response"
```

**If label is underrepresented after 200 examples:** Perform targeted collection from specific thread types:
- For ANALYSIS: tactical/strategy-focused threads
- For HOT_TAKE: controversial player debates
- For REACTION: live match threads

**Class balance constraint:** No label should exceed 70% of the dataset. If one does, collect more from underrepresented labels before training.

---

## 5. Evaluation Metrics & Reasoning

**Primary metrics:**
- **Overall Accuracy:** What % of test examples does the model classify correctly? (Baseline: 33% random guess on 3-class task)
- **Per-class F1 score:** Harmonic mean of precision and recall for each label. F1 matters more than accuracy alone because it catches both false positives and false negatives.
- **Confusion matrix:** Shows which labels are being confused and in what direction (e.g., "mostly confusing ANALYSIS for HOT_TAKE")

**Why these metrics?**

1. **Accuracy alone is insufficient**: If the dataset is ~30% ANALYSIS, 35% HOT_TAKE, 35% REACTION, a "predict HOT_TAKE always" baseline gets 35% accuracy. We need per-class metrics to see if the model *genuinely learned* distinctions.

2. **F1 over precision/recall alone**: A model could achieve high recall (catches all ANALYSIS posts) but low precision (also predicts ANALYSIS for many non-ANALYSIS posts). F1 punishes both extremes.

3. **Confusion matrix shows failure modes**: If the model confuses ANALYSIS and HOT_TAKE constantly but distinguishes REACTION well, that tells us where the boundary is weak.

---

## 6. Definition of Success

**Specific performance threshold for "good enough":**

- **Baseline (Groq llama-3.3-70b zero-shot):** Expected to achieve ~40-55% accuracy (it's a hard task for a general LLM with no task training).
- **Fine-tuned DistilBERT success criteria:**
  - Minimum 70% overall accuracy (clear win over baseline)
  - Minimum F1 ≥ 0.65 for each label (all labels are learnable)
  - No label should have F1 < 0.50 (indicator of broken definition or data quality issues)

**If these thresholds aren't met, investigate:**
- Label definitions are too vague (need to tighten)
- Class imbalance in training data (need rebalancing)
- Boundary between two labels is genuinely hard for the model (document as a limitation)

**Success criteria specific to r/soccer task:**
The classifier should be useful enough to:
- Automatically flag high-quality posts (ANALYSIS) in a moderation pipeline
- Help users filter for substantive discussion
- Detect when a debate is happening at different quality levels

If F1 for ANALYSIS ≥ 0.70, the model is reliable enough for this use case.

---

## 7. AI Tool Plan

### **A. Label Stress-Testing**
**Task:** Give the label definitions and 10 ambiguous r/soccer posts to Claude, ask it to classify them independently.

**Purpose:** If Claude confidently misclassifies posts I can classify cleanly, my definitions are still too vague.

**Process:**
1. Paste the 3 label definitions (from Section 2 above)
2. Paste 10 borderline posts (collected during reading phase)
3. Ask: "Classify each independently using the label definitions provided. Explain your reasoning for any you're uncertain about."
4. Compare Claude's classifications to my own
5. If disagreement on >30% of posts: revise label definitions before annotating 200 examples
6. If agreement on >70%: definitions are solid enough for annotation

**Disclosure:** Will note in README if this was done and how many posts were stress-tested.

---

### **B. Annotation Assistance (Optional)**
**Decision:** I will NOT use pre-labeling to speed up annotation. Reason: With only 200 examples and 3 labels, manual annotation (~2 hours) is fast enough. Pre-labeling introduces inconsistency risk and makes me lazy in reviewing.

**If I change my mind mid-project:**
- Use Claude to pre-label a batch
- Manually review and correct every single label (no skimming)
- Track which examples were pre-labeled in a separate column (for transparency)
- Report this in the AI usage section

---

### **C. Failure Analysis**
**Task:** After getting wrong predictions from the fine-tuned model, feed them to Claude.

**Purpose:** Spot systematic patterns I might miss when reading examples one-by-one.

**Process:**
1. Extract all misclassified test examples with true label and predicted label
2. Paste to Claude with this prompt: "These are posts my classifier got wrong. What patterns do you notice? (length, sarcasm, ambiguous language, specific label pairs that are confused, etc.)"
3. Claude identifies patterns
4. I manually re-read the identified examples and verify patterns are real
5. Document findings in the evaluation report (including what I had to correct or discard)

**Example findings to look for:** 
- "Model confuses ANALYSIS and HOT_TAKE when posts are <50 words"
- "All REACTION misclassifications involve sarcasm"
- "Model predicts ANALYSIS for technical posts even when they lack evidence"

**Disclosure:** Will note in README which patterns Claude suggested and which I verified.

---

## Checkpoint: Milestone 1 Complete ✓

- ✅ Chose community (r/soccer) with clear reasoning
- ✅ Read 20 representative posts and identified patterns
- ✅ Defined 3 mutually exclusive labels (ANALYSIS, HOT_TAKE, REACTION)
- ✅ Each label has: one-sentence definition, 2 concrete examples, non-examples
- ✅ Documented 4 hard edge cases with explicit decision rules
- ✅ Stress-tested definitions mentally against ambiguous posts
- ✅ Outlined data collection plan with balance targets
- ✅ Justified evaluation metrics (accuracy + per-class F1 + confusion matrix)
- ✅ Set specific success thresholds (70% accuracy, F1 ≥ 0.65 per label)
- ✅ Documented AI tool usage plan (stress-test, optional pre-label, failure analysis)

**Ready for Milestone 2:** Write spec of data collection process and evaluation plan (already done above; Milestone 2 will formalize this).

