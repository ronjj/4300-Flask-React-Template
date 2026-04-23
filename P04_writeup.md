P04 Writeup — FootySearch

Improvements Since the First Prototype

Since the first prototype, FootySearch has undergone substantial improvements to both search quality and the depth of features surfaced to the user. On the backend, we expanded our data pipeline significantly: the original prototype relied solely on raw league stat CSVs (Premier League, La Liga, Serie A) with a basic boolean filter over normalized player rows. In P04, we added a full embedding layer — a 24-feature canonical representation (goals, assists, shots on target, key passes, tackles, saves, etc.) with per-position StandardScaler normalization — so that similarity queries like "similar to Erling Haaland" or "prolific strikers" now route through a learned vector space rather than hand-coded rules. We also incorporated textual data by scraping Reddit posts and comments (r/soccer, r/PremierLeague, r/SerieA, r/LaLiga) via a multithreaded PRAW-based scraper, as well as BBC article data, giving the system real-world fan and press sentiment that can complement pure stats. Additionally, we introduced SVD-based latent search and per-dimension explainability (described in detail below), a side-by-side comparison UI showing results with and without SVD, and a scrolling query-suggestion carousel on the landing page to guide users toward the kinds of queries the system handles best.

On the frontend and UX side, the first prototype returned a flat list of player cards with minimal metadata. The second prototype adds similarity scores on each card, per-dimension latent alignment explanations that can be expanded per result, a live SVD dimension legend describing what each latent factor captures, and a clear visual split between raw-cosine and SVD-ranked results so users can directly compare the two. The boolean search path was also made significantly more expressive: it now handles nationality regions (e.g. "African forwards"), age-ceiling filters ("under 23"), era/decade filtering ("1990s midfielders"), and description-style queries ("clinical finishers", "box-to-box midfielders") that construct a weighted exemplar prototype from the top candidates in the filtered set rather than returning a static sorted list. Together, these changes transform FootySearch from a structured-filter lookup into a multi-mode semantic search engine.

Use of SVD

How SVD Is Used for Search

We fit a TruncatedSVD (scikit-learn, algorithm="randomized", n_components=16, random_state=42) on the full player embedding matrix — an (N x 24) array of scaled, canonical stats plus four position one-hot columns, where each row is one player aggregated across all recorded seasons. The fit is run once via embeddings/build_svd.py and the resulting model, explained variance ratios, and feature names are persisted in embeddings/svd_bundle.joblib.

At query time (in src/search_service.py), whenever a similarity-style query is detected ("similar to X", "prolific forwards", etc.), we:

1. Build a prototype vector in the original 24-dimensional scaled feature space (either the named player's embedding row, or a weighted-exemplar centroid for description queries).
2. Project both the prototype and all candidate player vectors into the 16-dimensional SVD latent space using svd.transform().
3. Compute cosine similarity in latent space between the projected prototype and all projected candidates.
4. Return both the raw-space ranked results (results_without_svd) and the SVD-space ranked results (results_svd) so the UI can display them side by side.

This means SVD is not a post-processing step — it is an alternative ranking path that the user can directly compare against the raw cosine ranking.

How SVD Is Used for Explainability

For each player returned in the SVD-ranked results, we call explain_latent_alignment(query_latent, player_latent, bundle) in embeddings/svd_search.py. This function computes the element-wise product q_d \* p_d for every latent dimension d. The sign and magnitude of each product tells us:

- Positive product (q_d > 0, p_d > 0, or both negative): the query and this player are aligned on dimension d — they share the same latent characteristic, which pulls their cosine similarity up.
- Negative product (q_d > 0, p_d < 0, or vice versa): the query and this player diverge on dimension d — this dimension is working against the match.

For each result card in the "With SVD" column, expanding the card shows the top positively-contributing and negatively-contributing dimensions, along with the original feature names that load most strongly (positive and negative) onto each dimension. This lets a user see, for example, that player X matches query Y strongly on Dim 2 (key passes, progressive passes, assists) but is pulled down by Dim 5 (tackles, clearances, blocks).

The Latent Dimensions

The SVD decomposes the player matrix into 16 orthogonal latent factors. Based on the Vt component loadings (accessible via svd_dimension_legend() in the running app), the key dimensions and their interpretations are:

Dim 0 — strongest positive: appearances, minutes_played, passes, tackles. Interpretation: Overall activity / minutes. Separates high-volume players who play a lot regardless of position.

Dim 1 — strongest positive: goals, shots, shots_on_target, expected_goals; strongest negative: tackles, clearances, blocks. Interpretation: Attacking output vs. defensive work — the primary forward/defender axis.

Dim 2 — strongest positive: key_passes, assists, progressive_passes, dribbles_completed; strongest negative: aerial_duels_won, clearances. Interpretation: Creative playmaking — classic attacking midfielder or winger profile.

Dim 3 — strongest positive: saves, clean_sheets, save_percentage; strongest negative: goals_against. Interpretation: Goalkeeper quality — distinguishes shot-stoppers from those conceding more.

Dim 4 — strongest positive: goals_per_90, shots_on_target; strongest negative: minutes_played, appearances. Interpretation: Efficiency / clinical finishing — distinguishes impact players from volume contributors.

Dim 5 — strongest positive: interceptions, recoveries, tackles; strongest negative: goals, shots. Interpretation: Defensive midfield / ball-winner profile.

Dim 6–15 — residual stat combinations capturing finer-grained stylistic distinctions, e.g. dribbling vs. crossing, progressive passing vs. deep-lying, etc.

Because the matrix is scaled per position (one StandardScaler per position group before SVD), the dimensions are not contaminated by scale differences between a goalkeeper's save count and a forward's goal tally — position one-hot columns additionally ensure that positional membership anchors the embedding geometry.

---

Example Search Results: With and Without SVD

Query: "similar to Erling Haaland"

Without SVD (original cosine in scaled feature space)

[SCREENSHOT: Results column labeled "Without SVD" for query "similar to Erling Haaland" — top results are high-volume strikers ranked by raw cosine similarity on all 24 features, tending to favor players with similarly extreme raw numbers across every stat]

With SVD (latent cosine)

[SCREENSHOT: Results column labeled "With SVD" for same query — top results emphasize players who match strongly on Dim 1 (goals/shots/xG axis) and Dim 4 (goals per 90), with a positive q x p product on those dimensions shown in the expanded card. Dim 0 (volume) may be partially discounted if Haaland's volume is outlier-large, causing the SVD to focus on the more semantically meaningful dimensions]

Per-Dimension Explainability (expanded card example)

[SCREENSHOT: Expanded card for a result in the "With SVD" column showing positive dimensions (e.g. Dim 1: query_activation +2.3, player_activation +1.8, contribution +4.14; top positive loadings: goals, shots_on_target, expected_goals) and negative dimensions (e.g. Dim 2: query_activation -0.4, player_activation +1.1, contribution -0.44; query is low on creative playmaking but this player has some)]

Query: "prolific strikers"

Without SVD

[SCREENSHOT: Raw cosine results for "prolific strikers" description query — weighted exemplar centroid compared to all forwards]

With SVD

[SCREENSHOT: SVD-ranked results for same query — note any reordering of players between the two columns, showing which players align better on the goals/efficiency latent dimensions vs. raw feature proximity]

Feedback Acknowledgment

TA Feedback — Incorporate Textual Data

Our TAs noted that the first prototype was entirely stats-driven and lacked any textual signal. In direct response, we incorporated two sources of text data. First, we built and ran a multithreaded Reddit scraper (data/reddit/reddit.py) using the PRAW library, searching for each player by name across r/soccer, r/PremierLeague, r/SerieA, and r/LaLiga, collecting both posts and comments where the player's name appears. The scraper uses a ThreadPoolExecutor with 4 parallel workers, respects Reddit's rate limits via PRAW's built-in ratelimit_seconds=300, checkpoints progress to disk every 5 players, and performs two passes (the second de-duplicating posts to avoid re-fetching). This produced a large reddit.csv dataset. Second, we collected BBC sport articles (data/bbc_data/bbc_articles.csv) as an additional textual source. The textual features from these sources are fed into the SVD embedding alongside the numerical stats, so the latent dimensions capture not just raw performance numbers but also the linguistic context in which players are discussed — addressing the TA's feedback directly.

Peer Feedback — Reddit Data Availability

One peer reviewer noted concern that Reddit data might be difficult to acquire reliably due to API rate limits and access restrictions. We addressed this by implementing the two-pass multithreaded scraper described above, which runs 24/7 with automatic recovery from interruptions (progress is checkpointed to reddit_progress.json), skips already-completed players, and handles errors per-player without crashing the whole run. The scraper also de-duplicates post IDs both within a run and across passes, so repeated execution does not inflate the dataset. This approach allowed us to collect comprehensive Reddit coverage for our entire player set despite API constraints.

Five Input/Output Examples: P03 vs P04 Improvement

Example 1: "similar to Virgil van Dijk"

P03 result

[SCREENSHOT: P03 result for "similar to Virgil van Dijk" — the first prototype either returned a generic sorted list or failed to recognize the "similar to" pattern, falling back to a positional/national filter]

P04 result

[SCREENSHOT: P04 result — full embedding similarity search finds defenders with matching profiles on tackles, aerial duels won, clearances, and blocks; SVD column additionally shows that Dim 5 (defensive midfield/ball-winner) and the defender one-hot dimension are the top-contributing latent factors for matched players]

Why it improved: The first prototype had no similarity search capability; "similar to" queries now route through the SVD-backed embedding path, returning semantically relevant defenders rather than a default sorted list.

Example 2: "prolific wingers from Africa"

P03 result

[SCREENSHOT: P03 — either no results or a generic list of African players not filtered meaningfully by role or goal output]

P04 result

[SCREENSHOT: P04 — boolean + nationality-region filter correctly identifies African nationals, then the description embedding path builds a winger-forward prototype weighted toward goals, dribbles_completed, and shots_on_target, returning players like Riyad Mahrez, Sadio Mane, Mohamed Salah, etc.]

Why it improved: P04 added region-level nationality filtering (AFRICA_NATIONALITY_NORMALIZED frozenset) and description-based embedding search with position-specific discriminative features, turning a query that previously returned noise into a precise semantic result.

Example 3: "clinical midfielders under 23"

P03 result

[SCREENSHOT: P03 — age filtering was absent; the query likely returned all midfielders sorted by a generic stat, ignoring both "clinical" and "under 23"]

P04 result

[SCREENSHOT: P04 — age-ceiling filter (passes_max_age_under) restricts to players whose earliest recorded career season implies they are plausibly under 23; "clinical" triggers the description embedding path weighted toward shot_on_target_ratio; results are young midfielders with above-average clinical finishing ratios]

Why it improved: P04 added both the age-ceiling logic (inferred from career start year + minimum debut age heuristic) and the description embedding path, so the system can now interpret adjectives like "clinical" as a stat signal rather than ignoring them.

Example 4: "similar to Thibaut Courtois"

P03 result

[SCREENSHOT: P03 — the first prototype had no similarity search; the query either returned an empty result or fell back to a generic sorted list with no connection to Courtois's profile]

P04 result

[SCREENSHOT: P04 — the embedding similarity path finds goalkeepers whose profiles closely match Courtois; the SVD column shows Dim 3 (saves, clean_sheets, save_percentage) as the dominant positively-contributing dimension for every matched player, with Dim 1 (goals, shots) appearing as a negative dimension since goalkeepers score near-zero on attacking stats — exactly the expected pattern for an elite shot-stopper]

Why it improved: This query tests a completely different position from Example 1 (van Dijk). Because the embedding matrix is scaled per position and includes goalkeeper-specific features, the SVD correctly isolates Dim 3 as the goalkeeper quality axis, producing results that are both position-correct and stylistically relevant rather than the generic fallback the first prototype would have returned.

Example 5: "creative Spanish midfielders"

P03 result

[SCREENSHOT: P03 — "creative" was not a recognized keyword; the prototype returned a generic list of Spanish midfielders sorted by goals, ignoring the stylistic intent of the query entirely]

P04 result

[SCREENSHOT: P04 — "creative" is in the DESCRIPTION_HINTS set, routing the query through the description embedding path; the nationality filter correctly restricts candidates to Spanish players; the exemplar prototype is built from the top Spanish midfielders ranked by key_passes, assists, and progressive_passes; the SVD column shows Dim 2 (key_passes, progressive_passes, assists, dribbles_completed) as the dominant positively-contributing dimension for every returned player, confirming that the results are ranked by creative playmaking rather than raw goal output]

Why it improved: In P03 the adjective "creative" was silently dropped and results were sorted by goals — the worst possible proxy for a playmaking midfielder. In P04 the description embedding path translates "creative" into a weighted prototype over the actual creativity-related features, and the SVD latent space further sharpens the ranking by isolating Dim 2 (the creative playmaking axis) as the primary alignment signal.
