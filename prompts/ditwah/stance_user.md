Article Title: {{article_title}}

Article Content:
{{article_content}}

Hypothesis: {{hypothesis}}

Analyze the article's stance towards the hypothesis. Respond with JSON:
{
  "agreement_score": <float between -1.0 (strongly disagree) and 1.0 (strongly agree)>,
  "confidence": <float between 0.0 and 1.0>,
  "stance": "<one of: strongly_agree, agree, neutral, disagree, strongly_disagree>",
  "reasoning": "<2-3 sentence explanation of your assessment>",
  "supporting_quotes": ["<relevant quote 1>", "<relevant quote 2>"]
}

Important:
- If the article doesn't mention the hypothesis topic at all, use stance "neutral" with confidence < 0.3
- Only use "strongly_agree" or "strongly_disagree" if the article explicitly and clearly takes that position
- Keep quotes short (max 200 chars each), extract at most 3 quotes
- Reasoning should be concise and evidence-based
