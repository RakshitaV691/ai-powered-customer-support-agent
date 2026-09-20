# Decision Log

## 1. Brand selection
Selected SpotifyCares as the single brand for the support agent so that the historical examples and reply generation remain focused on one support context.

## 2. Intent taxonomy
Defined seven intents: Playback, Technical/App Issue, Content Availability, Subscription/Payment, Ads/Promotions, Feature Request, and Other/General.

## 3. Separate intent and action
Kept intent classification separate from Auto-handle vs Escalate because the same intent can sometimes require different handling depending on whether the issue is account-specific or requires human intervention.

## 4. Escalation policy
Escalate account-specific issues, payment/billing problems, refunds or charge disputes, requests requiring private account information, unclear requests, sensitive situations, and unresolved issues requiring human troubleshooting.

## 5. Golden set size
Created a 200-example hand-labelled golden set, which is within the required 150–250 example range.

## 6. Golden set balance
Used a roughly balanced distribution across the seven intents so that evaluation is not dominated by one class.

## 7. Duplicate removal
Removed duplicate customer messages from the golden set to avoid evaluating repeated examples as independent evidence.

## 8. Historical data size
Used a 1,000-pair Spotify historical subset instead of the full dataset to keep retrieval and experimentation practical and reproducible.

## 9. Retrieval method
Used TF-IDF with cosine similarity to retrieve historically similar customer messages and their replies.

## 10. Retrieval leakage prevention
Excluded an exact matching historical customer message from retrieval when evaluating that message, preventing the system from simply retrieving the same example.

## 11. Majority baseline
Added a majority-class baseline to establish a simple lower-bound reference for intent classification.

## 12. Rule-based baseline
Added a keyword/rule-based classifier as a stronger simple baseline before comparing against the LLM-based agent.

## 13. Evaluation metric
Used macro-averaged precision, recall, and F1 in addition to accuracy because the task contains multiple intent classes and performance should be considered across all classes.

## 14. LLM selection
Used Google's Gemini API through the available Free Tier so the project can be developed and evaluated without paid API usage.

## 15. Historical grounding
Designed reply generation to use retrieved historical examples as evidence rather than generating unsupported account-specific actions or information.