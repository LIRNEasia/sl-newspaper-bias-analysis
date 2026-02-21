Analyze these {{article_count}} news articles about Cyclone Ditwah and identify {{num_claims}} specific, verifiable claims made across the coverage.

Articles:
{{articles_json}}

Instructions:
1. Identify {{num_claims}} key claims or statements that appear across multiple articles
2. Each claim should be:
   - Specific and verifiable (not vague or general)
   - Mentioned or implied by at least 3 articles
   - Significant to understanding the cyclone's impact or response
3. Categorize each claim using these categories: {{categories_list}}
4. Prioritize claims that show variation across sources (some agree, some disagree)

Return a JSON array of claims with this structure:
[
  {
    "claim_text": "The exact claim or statement",
    "claim_category": "one of the categories above",
    "confidence": 0.9
  }
]

Return ONLY the JSON array, no other text.
