# Availability Rules
- Common colors: black, white, beige, navy, pink, red, green, brown
- Sizes: XXS–XXL or numeric 0–18
- If both color and size given → return stock availability
- If out of stock → suggest restock notification or alternative

## Parsing & Normalization Rules (for chat extraction)
- Accept color/size even if user says them in free text and mixed order.
- Color normalization:
  - Map synonyms (e.g., “navy blue” → “navy”, “cream white” → “cream”, “khaki/olive-green” → “olive”).
  - Fuzzy match within Levenshtein distance ≤2 (e.g., “gren” → “green”).
- Size normalization:
  - Map letter ↔ number (if known per category) and expand words: “extra small” → “XS”.
  - Accept ranges (“S or M”) and ask a nudge: “Which one should I check first?”.
- Product inference:
  - If user says “city jacket” or “office blazer” → candidate: “jacket”.
  - When ambiguous, ask: “Do you mean our Everyday Jacket or another jacket?”

## Low-Stock Recommendation Logic
- If user’s exact {product}/{color}/{size} is AVAILABLE: answer “5+ in stock”.
- Immediately follow with cross-sell if any sibling color has <3: “FYI, {other_color} is running low.”
- If user says “yes” to low-stock alternatives: show last-season similar item in SAME color with “final 2 left”.

## Restock Notifications
- If OUT OF STOCK:
  - Say: “Restock in about 2 weeks. Prefer email or SMS for a ping?”
  - Store contact preference and product key: `{product}:{color}:{size}`.
  - Confirm: “I’ll notify you as soon as it lands.”
