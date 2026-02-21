Analyze whether each article agrees, disagrees, or remains neutral about this claim:

Claim: "{{claim_text}}"

Articles:
{{articles_json}}

For each article, determine:
1. Does it agree, disagree, or remain neutral about the claim?
2. How confident are you? (0.0 to 1.0)
3. What is your reasoning?
4. What quotes support your assessment? (up to 2 quotes)

Return a JSON array with this structure:
[
  {
    "article_id": "uuid",
    "stance_score": 0.7,  // -1.0 (strongly disagree) to +1.0 (strongly agree), 0 = neutral
    "stance_label": "agree",  // one of: strongly_agree, agree, neutral, disagree, strongly_disagree
    "confidence": 0.9,
    "reasoning": "Brief explanation of the stance",
    "supporting_quotes": ["quote 1", "quote 2"]
  }
]

Guidelines:
- stance_score: -1.0 to -0.6 = strongly_disagree, -0.6 to -0.2 = disagree, -0.2 to 0.2 = neutral, 0.2 to 0.6 = agree, 0.6 to 1.0 = strongly_agree
- If the article doesn't mention the claim, mark as neutral with low confidence
- Focus on what the article explicitly states, not implications

Return ONLY the JSON array, no other text.
