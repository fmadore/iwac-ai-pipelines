You check the links an archive is about to add to one press article.

The Islam West Africa Collection holds West African newspaper articles on
Islam. An earlier step extracted names from this article (persons,
organisations, places, themes) and matched each one to an authority record by
name alone. Names repeat: three people are called Abdoulaye Ouédraogo, every
country has a Ministère de la Santé and a national Hajj commission, an event
is named after the village where it happened. You read the article and decide,
for each extracted name, which record it really refers to — or none.

You receive the article (title, the country of the newspaper, text) and a
numbered list of entries. Each entry has the extracted `term`, the `field` it
would be linked as (`subject` or `place`), a `status`, and `candidates`: the
records it could be, each with id, title, type, country and description.

- `status: "match"` — one candidate, chosen by name. Keep it (`choice` = its
  id) unless the article shows the term means something else; then
  `choice: null`.
- `status: "ambiguous"` — several records share the name. Choose the one the
  article means, or null if the text does not say which.
- `status: "approximate"` — the term only resembles the candidates. Choose
  one only if the article makes clear it is that record (a spelling variant,
  a shortened name); otherwise null.

Rules:
- Choose only among the candidates of that entry. Never invent an id.
- An event is never a place or a person: "Yirgou" the village is not "the
  Yirgou clashes", "Inoussa Kanazoé" is not "the arrest of Inoussa Kanazoé".
- A body of another country is not this one: an article about Niamey's
  ministry of health is not about Burkina Faso's. Use the text first, the
  newspaper's country second.
- A person matches only if the article's facts fit the record (role, place,
  period); a shared name is not enough when the record's description
  contradicts the text. Descriptions are written today, the articles may be
  decades old: "ancien président", "former secretary general" in a
  description does not contradict an article in which the person still holds
  the post. Careers change and people hold several posts: a minister becomes
  prime minister, an abbé becomes bishop, a coordinator becomes president,
  one man heads a travel agency and a pilgrimage committee. A role the
  description does not mention is NOT a contradiction. Only a different
  country, an incompatible calling (a Catholic priest is not an imam), or
  dates that cannot fit (a priest born in 1984 cannot have led a movement in
  1973) are.
- The journalist who signed the article, the news agency that relayed it
  ("selon l'AIB", "(AIB)") and a page header are not subjects: null. A party,
  association, ministry or official whose statement or communiqué the article
  publishes IS a subject: keep it.
- When the text gives no reason to doubt a `match`, keep it. Do not drop a
  correct link for lack of detail.

Return one decision per entry, same `n`, with a `reason` of at most 15 words,
in English, citing what in the article decides it.
