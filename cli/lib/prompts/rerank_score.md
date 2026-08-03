"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {title} - {document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

CRITICAL INSTRUCTION: Rate 0-10 (10 = perfect match).
You must output ONLY the raw digit/number (e.g., 10 or 8.5). 
Do NOT include the word "Score:", do NOT include any punctuation, explanation, or markdown formatting.

Score:
"""