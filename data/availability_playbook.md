# Availability Playbook
1. Ask product type (e.g., jacket, skirt, blouse)
2. Ask color + size
3. If available: "We have 5+ in stock."
4. If limited: "Only 2 left in stock."
5. If sold out: "Currently out of stock. Restock in 2 weeks. Want a notification?"

## Edge Cases
- User gives only color or only size → acknowledge what you have and ask for the missing slot.
- Unknown color word:
  - Try normalization; if still unknown: “I may be off—did you mean one of these: {top_matches}?”
- “Any color is fine”:
  - Pick the color with highest stock first; if tie, pick collection default (e.g., Black for Tailored Pants).
- “Any size”:
  - Offer closest two sizes around typical M; ask body reference or brand size table link.

## Ready-to-say Responses
- AVAILABLE: “We have **5+** in stock for the {product} in {color}/{size}.”
- LOW: “That exact pick is available, and **{other_color}** is **low in stock** right now.”
- OOS: “{product} in {color}/{size} is out of stock. **Restock in ~2 weeks.** Want a notification by email or SMS?”
