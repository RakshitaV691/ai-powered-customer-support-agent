# AI Support Agent for SpotifyCares

## Project Overview

This project implements an AI-powered customer support agent for SpotifyCares using historical customer-support conversations.

The agent performs four main tasks:

1. Classifies each customer message into a predefined support intent.
2. Decides whether the request should be Auto-handled or Escalated.
3. Provides an escalation reason when human intervention is required.
4. Generates a concise customer-support reply grounded in historically similar conversations.

The project also compares the AI agent against simple baseline approaches to measure whether the AI-based approach provides meaningful improvement.

## Intent Taxonomy

The agent uses seven support intents:

| Intent | Description |
|---|---|
| Playback | Problems with playing, pausing, skipping, shuffle, or repeat |
| Technical/App Issue | App crashes, installation, device, version, settings, or other technical problems |
| Content Availability | Songs, albums, or other content that is missing or unavailable |
| Subscription/Payment | Premium, subscriptions, payments, billing, refunds, or charges |
| Ads/Promotions | Advertising, unwanted ads, promotional offers, or promotions |
| Feature Request | Requests or suggestions for new or changed functionality |
| Other/General | Unclear requests, acknowledgements, or messages that do not fit another intent |

## Problem Framing

For Spotify customer support, a good agent should identify the customer's intent, provide a reply grounded in historically observed SpotifyCares responses, and route cases to a human when the issue is account-specific, sensitive, unclear, or requires actions the agent should not perform automatically.

The goal of this project is to demonstrate a reproducible support-agent pipeline rather than build a production-ready customer service system.

### What I Chose Not to Build

- A production deployment or live customer-facing system.
- Direct access to customer accounts or private information.
- Automated refunds, billing changes, or other account-level actions.
- Full multi-turn conversation management across the entire Twitter dataset.
- Evaluation on the full multi-million-row dataset.

## Handling Policy

The agent separates intent classification from the handling decision.

### Auto-handle

The agent can auto-handle common:

- Playback problems
- App/device troubleshooting
- Content availability questions
- General feature requests
- Ads and promotions questions

### Escalate

The agent escalates when the request involves:

- Account-specific issues
- Payment or billing problems
- Refunds or charge disputes
- Private account information
- Unclear customer requests
- High-risk or sensitive situations
- Issues requiring human troubleshooting
- Feature requests requiring human review
- Unresolved issues with insufficient context

The escalation reason is selected from a predefined set of reasons rather than being generated as unrestricted text.

## Dataset

The project uses the Customer Support on Twitter dataset.

For this implementation:

- The brand selected was SpotifyCares.
- Historical Spotify customer-support conversations were extracted from the dataset.
- A working set of 1,000 customer-message/reply pairs was used for historical retrieval.
- A 200-example hand-labelled golden set was created for evaluation.
- Duplicate customer messages were removed from the golden set.

The golden set contains examples from all seven intent categories and includes the expected handling action and escalation reason where applicable.

## System Approach

The system uses two main components:

### 1. Historical retrieval

Historical SpotifyCares conversations are converted into TF-IDF vectors.

For a new customer message:

1. The message is converted into a TF-IDF vector.
2. Cosine similarity is calculated against the historical customer messages.
3. The top three most similar historical examples are retrieved.
4. Their historical replies are used as context for reply generation.

During evaluation, an exact matching historical customer message is excluded from retrieval to reduce evaluation leakage.

### 2. Gemini-based agent

Google Gemini is used for:

- Intent classification
- Handling decision
- Escalation reason selection
- Customer reply generation

The generated reply is instructed to remain grounded in the retrieved historical examples and avoid inventing account-specific information.

## Setup and Usage

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the evaluation pipeline:

```bash
python evaluate.py
```

## Baselines

Two simple baselines were implemented to provide reference points for the AI agent.

### Majority Baseline

The majority baseline predicts the most frequent intent in the golden set for every customer message.

### Rule-Based Baseline

The rule-based baseline uses manually defined keywords associated with each intent. If no relevant keyword is found, it predicts `Other/General`.

The baselines provide a simple comparison against the Gemini-based agent.

## Baseline Results

The current baseline results on the 200-example golden set are:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority Baseline | 0.2500 | 0.0357 | 0.1429 | 0.0571 |
| Rule-Based Baseline | 0.4250 | 0.7361 | 0.3976 | 0.3915 |
| Gemini Agent | 0.6667 | 0.5833 | 0.5833 | 0.5556 |

The rule-based baseline correctly classified 85 out of 200 examples.

The Gemini-based agent was evaluated on 9 saved examples because live API evaluation was limited by Gemini service availability. The results are reported separately from the full 200-example baseline evaluation.

## What Is Misleading About My Headline Number?

The Gemini agent achieved 66.67% accuracy on the saved evaluation sample. However, this number should not be interpreted as production-level accuracy or as a direct comparison with the baselines. The Gemini result is based on only 9 saved examples because live API evaluation was limited by Gemini service availability, while the majority and rule-based baselines were evaluated on the full 200-example golden set. The small Gemini sample therefore provides an initial signal rather than a statistically strong estimate of generalization.

## Evaluation

A hand-labelled golden set of 200 customer messages is used for evaluation.

For intent classification, the following metrics are reported:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

Macro-averaged metrics are included so that performance across all seven intent classes is considered rather than relying only on the overall accuracy.

The evaluation reports the two baselines on the full 200-example golden set and reports Gemini agent metrics separately on 9 saved evaluation examples. Because the evaluation sample sizes differ, the results should not be interpreted as a direct full-dataset comparison.

In addition to automated intent metrics, an LLM-as-judge evaluation was performed on 2 generated replies to assess historical grounding, relevance/helpfulness, unsupported claims, and escalation handling. The two judged replies were also assessed manually using the same criteria. The LLM judge and human scores were 8/8 and 7/8 for both examples, respectively.

## Failure Analysis

The Gemini agent correctly classified 6 of the 9 saved evaluation examples and incorrectly classified 3.

Top failure modes observed:

1. **Short or conversational Content Availability messages**
   - Example: A customer acknowledgement related to content availability was classified as Other/General.
   - Hypothesis: The message did not contain enough explicit content-related keywords for the model to identify the underlying intent.

2. **Ambiguous Playback vs. Content Availability**
   - Example: A Playback-related message discussing songs in the user's library was classified as Content Availability.
   - Hypothesis: References to songs and library content can overlap between playback problems and availability questions.

3. **Indirect Subscription/Payment messages**
   - Example: A short acknowledgement related to a subscription/payment conversation was classified as Other/General.
   - Hypothesis: The customer's message alone did not clearly state the underlying billing or subscription issue.

4. **General messages with insufficient intent evidence**
   - Example: Very short messages such as acknowledgements can lack enough information to distinguish the underlying support intent.
   - Hypothesis: The agent primarily sees the current customer message, so important context from earlier turns may be unavailable.

5. **Context-dependent intent boundaries**
   - Example: Messages containing general references to songs, playlists, or account activity can be mapped to different intents depending on the surrounding conversation.
   - Hypothesis: Several intent categories share vocabulary, so additional conversational context and stronger retrieval could improve classification.

These failures suggest that the main weakness is not a single intent category, but the difficulty of classifying short, indirect, and context-dependent customer messages.

## Limitations

- The Gemini agent evaluation contains only 9 saved examples because live API evaluation was limited by Gemini service availability.
- The baseline models were evaluated on the full 200-example golden set, so their metrics are not directly comparable with the Gemini results.
- The LLM-as-judge evaluation covers only 2 generated replies and should be treated as a small qualitative sample rather than a statistically reliable estimate of overall reply quality.
- Human assessment was also performed on only those 2 replies.
- The historical retrieval set contains 1,000 SpotifyCares conversation pairs rather than the full dataset.
- Intent classification can be difficult for short, ambiguous, or context-dependent customer messages.
- Reply quality and intent-classification accuracy are evaluated separately; a correct intent prediction does not necessarily guarantee a high-quality customer reply.

For implementation decisions, see [Decision Log](decision_log.md).

## Future Improvements

- Expand the Gemini evaluation set beyond the current saved sample and evaluate it on a larger, consistently sampled subset.
- Improve intent classification for short and ambiguous customer messages using stronger intent-specific examples and retrieval context.
- Improve historical retrieval so that the selected past conversation is more closely matched to the customer’s issue.
- Expand human evaluation of generated replies and measure agreement with the LLM judge on a larger sample.
- Test the escalation policy more thoroughly on edge cases involving billing, account-specific issues, and unclear requests.