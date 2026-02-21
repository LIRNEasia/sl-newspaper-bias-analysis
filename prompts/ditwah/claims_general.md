You are analyzing {{claim_count}} similar claims from different articles about Cyclone Ditwah.
These claims come from sources: {{sources_list}}

Individual claims:
{{claims_list}}
{{overflow_text}}
Generate ONE general claim that captures the common theme across these individual claims.

The general claim should:
- Capture the essence of what these claims are saying
- Be specific enough to be meaningful
- Be general enough to cover all the individual claims
- Be 1-2 sentences
- Be verifiable

Also categorize the claim using one of these categories: {{categories_list}}

Return ONLY a JSON object:
{"claim_text": "Your general claim here", "claim_category": "category_name"}

Return ONLY the JSON, no other text.
