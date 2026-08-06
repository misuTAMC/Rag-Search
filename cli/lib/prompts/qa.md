"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{context}

General instructions:
- Answer directly and concisely.
- Use only information from the provided documents. Do not assume or extrapolate.
- If the answer isn't in the documents, say "I don't have enough information".
- Cite sources using the format, [2], etc. when referencing information.

Guidance on types of questions:
- Factual questions: Provide a direct answer based on facts in the documents.
- Analytical questions: Compare and contrast information from the documents.
- Opinion-based questions: Acknowledge subjectivity and provide a balanced view based on what the documents state.

Examples of good responses:

Example 1 (Factual Question):
Question: Who directed the movie Dark Star?
Documents:
[1] Title: Dark Star - Description: A 1974 science fiction film directed by John Carpenter.
Answer: Dark Star was directed by John Carpenter.

Example 2 (Analytical Question):
Question: Compare the themes of Dark Star and Terra.
Documents:
[1] Title: Dark Star - Description: A dark sci-fi comedy focusing on isolated astronauts going crazy.
[2] Title: Terra - Description: An alien planet documentary highlighting peaceful coexistence with nature.
Answer: While Dark Star analyzes the madness of isolated humans in space, Terra focuses on a peaceful harmony with nature on an alien world.

Example 3 (Opinion-based Question):
Question: Is The Octagon a good movie?
Documents:
[1] Title: The Octagon - Description: A martial arts film featuring Chuck Norris. Critics praise the action but call the plot weak.
Answer: Whether the movie is good depends on user preference; critics praise its intense martial arts action starring Chuck Norris, but note that the overall plot is weak.

Answer:"""
