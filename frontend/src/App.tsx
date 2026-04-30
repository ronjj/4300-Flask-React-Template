import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useScroll, useTransform } from "framer-motion";
import ReactMarkdown from "react-markdown";
import "./App.css";
import Logo from "./components/Logo";
import SearchBar from "./components/SearchBar";
import PlayerGrid from "./components/PlayerGrid";
import { PlayerCardData, PlayerChatResponse, PlayerStats, SearchResponse } from "./types";
import POPULAR_PLAYERS from "./data/popularPlayers";
import PlayerProfile from "./components/PlayerProfile";
import searchSvg from "./assets/search.svg";
import soccerballSvg from "./assets/soccerball.svg";
import compassSvg from "./assets/compass.svg";

const EXAMPLE_QUERIES = [
  "best brazilian wingers",
  "top forwards in La Liga",
  "fastest defenders in the Premier League",
  "creative midfielders from Argentina",
  "most assists in Serie A",
  "box-to-box midfielders",
  "best Spanish goalkeepers",
  "prolific wingers from Africa",
];

function QueryCarousel({ onSelect }: { onSelect: (q: string) => void }): JSX.Element {
  const trackRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number | null>(null);
  const posRef = useRef(0);
  const pausedRef = useRef(false);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;

    const speed = 0.25;

    const step = () => {
      if (!pausedRef.current) {
        posRef.current += speed;
        const halfWidth = track.scrollWidth / 2;
        if (posRef.current >= halfWidth) posRef.current -= halfWidth;
        track.style.transform = `translateX(-${posRef.current}px)`;
      }
      animRef.current = requestAnimationFrame(step);
    };

    animRef.current = requestAnimationFrame(step);
    return () => {
      if (animRef.current !== null) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const chips = [...EXAMPLE_QUERIES, ...EXAMPLE_QUERIES];

  return (
    <div
      className="query-carousel-wrapper"
      onMouseEnter={() => {
        pausedRef.current = true;
      }}
      onMouseLeave={() => {
        pausedRef.current = false;
      }}
    >
      <div className="query-carousel-track" ref={trackRef}>
        {chips.map((q, i) => (
          <button
            key={`${q}-${i}`}
            className="query-chip"
            onClick={() => onSelect(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

type SearchStatus = "idle" | "loading" | "populated" | "empty" | "error";
type SearchMode = string | null;

function toCardData(results: PlayerStats[]): PlayerCardData[] {
  return results.map((player, index) => ({
    key: `${player.name}-${player.team ?? "unknown"}-${player.league ?? "unknown"}-${index}`,
    rank: index + 1,
    name: player.name,
    team: player.team,
    position: player.position,
    nationality: player.nationality,
    goals: player.goals,
    appearances: player.appearances,
    image: player.image,
    fullStats: player,
  }));
}

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean>(true);
  const [aiMode, setAiMode] = useState<boolean>(false);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiMeta, setAiMeta] = useState<{
    rewrittenQuery?: string | null;
    retrievalConfidence?: number | null;
    results?: import("./types").PlayerChatResult[];
    evidence?: import("./types").PlayerChatEvidence[];
    warnings?: string[];
  } | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [players, setPlayers] = useState<PlayerCardData[]>([]);
  const [playersSvd, setPlayersSvd] = useState<PlayerCardData[]>([]);
  const [svdAvailable, setSvdAvailable] = useState<boolean>(false);
  const [showSvdRanking, setShowSvdRanking] = useState<boolean>(false);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [searchMode, setSearchMode] = useState<SearchMode>(null);
  const [heroMode, setHeroMode] = useState<boolean>(false);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerCardData | null>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  const spotlightRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const loadConfig = async (): Promise<void> => {
      try {
        const response = await fetch(`${API_BASE}/api/config`);
        if (!response.ok) return;
        const data: { use_llm?: boolean } = await response.json();
        setUseLlm(Boolean(data.use_llm));
      } catch {
      }
    };
    void loadConfig();
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedPlayer) setSelectedPlayer(null);
        else if (heroMode) setHeroMode(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [heroMode, selectedPlayer]);

  useEffect(() => {
    document.body.style.overflow = (heroMode || selectedPlayer !== null) ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [heroMode, selectedPlayer]);

  const { scrollY } = useScroll();
  const vh = window.innerHeight;
  const heroLogoScale   = useTransform(scrollY, [0, vh * 0.65], [1, 0.26]);
  const heroLogoOpacity = useTransform(scrollY, [vh * 0.38, vh * 0.65], [1, 0]);
  const heroLogoX       = useTransform(scrollY, [0, vh * 0.65], [0, -70]);
  const headerLogoOpacity = useTransform(scrollY, [vh * 0.5, vh * 0.85], [0, 1]);

  useEffect(() => {
    const el = spotlightRef.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const { left, top } = el.getBoundingClientRect();
      el.style.setProperty("--sx", `${e.clientX - left}px`);
      el.style.setProperty("--sy", `${e.clientY - top}px`);
    };
    const onLeave = () => {
      el.style.setProperty("--sx", "-9999px");
      el.style.setProperty("--sy", "-9999px");
    };
    el.addEventListener("mousemove", onMove);
    el.addEventListener("mouseleave", onLeave);
    onLeave();
    return () => {
      el.removeEventListener("mousemove", onMove);
      el.removeEventListener("mouseleave", onLeave);
    };
  }, []);

  const scrollToShell = (): void => {
    shellRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const runSearch = async (term: string): Promise<void> => {
    const trimmed = term.trim();
    if (trimmed === "") {
      setPlayers([]);
      setPlayersSvd([]);
      setSvdAvailable(false);
      setShowSvdRanking(false);
      setStatus("idle");
      return;
    }
    scrollToShell();
    setStatus("loading");
    setSearchMode(null);
    setAiAnswer(null);
    setAiMeta(null);
    setSvdAvailable(false);
    setShowSvdRanking(false);
    try {
      const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(trimmed)}`);
      if (!response.ok) {
        setPlayers([]);
        setPlayersSvd([]);
        setSvdAvailable(false);
        setShowSvdRanking(false);
        setStatus("error");
        return;
      }
      const data: SearchResponse = await response.json();
      const standardResults = Array.isArray(data.results_without_svd)
        ? (data.results_without_svd as PlayerStats[])
        : (Array.isArray(data.results) ? data.results : []);
      const svdResults = Array.isArray(data.results_svd) ? (data.results_svd as PlayerStats[]) : [];

      // Build rank-delta map: name → (rank_without_svd − rank_with_svd)
      // positive = SVD moved the player higher in the list
      const rankDeltaMap = new Map<string, number>();
      if (data.svd_available && standardResults.length > 0 && svdResults.length > 0) {
        const withoutRanks = new Map(
          standardResults.map((p, i) => [p.name, i])
        );
        svdResults.forEach((player, svdIdx) => {
          const withoutIdx = withoutRanks.get(player.name);
          if (withoutIdx !== undefined) {
            rankDeltaMap.set(player.name, withoutIdx - svdIdx);
          }
        });
      }

      const nextPlayers = toCardData(standardResults).map((card) =>
        rankDeltaMap.has(card.name)
          ? { ...card, svdRankDelta: rankDeltaMap.get(card.name)! }
          : card
      );
      setPlayers(nextPlayers);
      const hasSvd = Boolean(data.svd_available && svdResults.length > 0);
      setSvdAvailable(hasSvd);
      if (hasSvd) {
        const svdCards = toCardData(svdResults).map((card) =>
          rankDeltaMap.has(card.name)
            ? { ...card, svdRankDelta: rankDeltaMap.get(card.name)! }
            : card
        );
        setPlayersSvd(svdCards);
      } else {
        setPlayersSvd([]);
      }
      setStatus(nextPlayers.length > 0 ? "populated" : "empty");
      setSearchMode(data.results[0]?.search_mode ?? null);
    } catch {
      setPlayers([]);
      setPlayersSvd([]);
      setSvdAvailable(false);
      setShowSvdRanking(false);
      setStatus("error");
    }
  };

  const runAiSearch = async (term: string): Promise<void> => {
    const trimmed = term.trim();
    if (!trimmed) return;
    scrollToShell();
    setStatus("loading");
    setAiAnswer(null);
    setAiMeta(null);
    setSearchMode(null);
    try {
      const response = await fetch(`${API_BASE}/api/player-chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
      });
      const data: PlayerChatResponse = await response.json();
      if (!response.ok) {
        setPlayers([]);
        setStatus("error");
        return;
      }
      const answer = data.answer || null;
      const meta = {
        rewrittenQuery: data.rewritten_query ?? null,
        retrievalConfidence: data.retrieval_confidence ?? null,
        results: Array.isArray(data.results) ? data.results : [],
        evidence: Array.isArray(data.evidence) ? data.evidence : [],
        warnings: Array.isArray(data.warnings) ? data.warnings : [],
      };
      const queryForGrid =
        (typeof data.rewritten_query === "string" && data.rewritten_query) ||
        (Array.isArray(data.results) && data.results[0]?.player_name) ||
        trimmed;
      await runSearch(queryForGrid);
      // Set after runSearch so it doesn't get cleared by runSearch's reset
      if (answer) setAiAnswer(answer);
      setAiMeta(meta);
    } catch {
      setPlayers([]);
      setStatus("error");
    }
  };

  const activeFetchId = useRef(0);

  const normalizeName = (s: string): string =>
    s.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase().trim();

  const handleFullStatsClick = (player: PlayerCardData): void => {
    setSelectedPlayer(player);
    if (!player.key.startsWith("popular-")) return;

    const id = ++activeFetchId.current;

    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(player.name)}`);
        if (!res.ok || id !== activeFetchId.current) return;
        const data: SearchResponse = await res.json();
        if (id !== activeFetchId.current) return;
        if (!Array.isArray(data.results) || data.results.length === 0) return;

        const lastName = normalizeName(player.name.split(" ").pop()!);
        const match = data.results.find((r: PlayerStats) => normalizeName(r.name).includes(lastName));
        if (!match) return;

        const enriched = toCardData([match])[0];
        setSelectedPlayer((prev) =>
          prev?.key === player.key
            ? { ...prev, fullStats: enriched.fullStats, image: enriched.image ?? prev.image }
            : prev
        );
      } catch {}
    })();
  };

  const shellMode = useMemo<"home" | "results">(() => {
    if (players.length > 0 || status === "loading" || status === "empty" || status === "error") return "results";
    return "home";
  }, [players.length, status]);


  const statusText =
    status === "loading"
      ? "Searching..."
      : status === "empty"
        ? "No results found."
        : status === "error"
          ? "Could not load results. Please try again."
          : null;

  const handleSearchChange = (nextValue: string): void => {
    setSearchTerm(nextValue);
    if (status !== "idle") setStatus("idle");
  };

  return (
    <>
      <div className={`full-body-container ${useLlm ? "llm-mode" : ""}`}>
        <motion.main
          className="welcome"
          ref={spotlightRef}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.22, ease: "easeOut" }}
        >
          <div className="welcome-spotlight" />
          <div className="welcome-content">
            <motion.div
              style={{
                scale: heroLogoScale,
                opacity: heroLogoOpacity,
                x: heroLogoX,
                originX: 0,
                originY: 0,
              }}
            >
              <Logo className="logo-hero" />
            </motion.div>
            <p className="tagline">World Class Results, Every Time.</p>
          </div>
        </motion.main>

        <motion.main
          className="app-shell"
          ref={shellRef}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          <header className="app-header">
            <div className="header-inner">
              <motion.div style={{ opacity: headerLogoOpacity }}>
                <Logo className="logo-header" />
              </motion.div>
              <div className="header-search">
                {!heroMode ? (
                  <motion.div layoutId="main-search">
                    <SearchBar
                      value={searchTerm}
                      inputRef={searchInputRef}
                      onFocus={() => { scrollToShell(); setHeroMode(true); }}
                      onChange={handleSearchChange}
                      onSubmit={() => aiMode ? void runAiSearch(searchTerm) : void runSearch(searchTerm)}
                      placeholder="look up the best Brazilian wingers..."
                      showAiToggle={useLlm}
                      aiMode={aiMode}
                      onAiToggle={() => setAiMode(m => !m)}
                    />
                  </motion.div>
                ) : (
                  <div className="search-bar-ghost" aria-hidden="true" />
                )}
              </div>
            </div>
          </header>

          <section className="content">
            <AnimatePresence mode="wait" initial={false}>
              {shellMode === "home" ? (
                <motion.div
                  key="home"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.22, ease: "easeOut" }}
                >
                  <div className="feature-tiles">
                    <div className="feature-tile">
                      <img src={searchSvg} alt="" aria-hidden="true" className="tile-icon" />
                      <h3 className="tile-title">Search</h3>
                      <p className="tile-subtitle">Type a player name or describe what you're looking for</p>
                    </div>
                    <div className="feature-tile">
                      <img src={soccerballSvg} alt="" aria-hidden="true" className="tile-icon" />
                      <h3 className="tile-title">Discover</h3>
                      <p className="tile-subtitle">Get ranked results with key stats</p>
                    </div>
                    <div className="feature-tile">
                      <img src={compassSvg} alt="" aria-hidden="true" className="tile-icon" />
                      <h3 className="tile-title">Explore</h3>
                      <p className="tile-subtitle">Dive into full player profiles</p>
                    </div>
                  </div>

                  <div className="popular-section">
                    <h2 className="section-title">Popular Players</h2>
                    <PlayerGrid players={POPULAR_PLAYERS} onFullStatsClick={handleFullStatsClick} />
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                >
                  <AnimatePresence>
                    {statusText && (
                      <motion.p
                        className="search-feedback"
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.18, ease: "easeOut" }}
                      >
                        {statusText}
                      </motion.p>
                    )}
                  </AnimatePresence>

                  <AnimatePresence>
                    {aiAnswer && status === "populated" && (
                      <motion.div
                        className="ai-answer-card"
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.26, ease: "easeOut" }}
                      >
                        <span className="ai-answer-icon">✦</span>
                        <div className="ai-answer-body">
                          <div className="ai-answer-text">
                            <ReactMarkdown>{aiAnswer}</ReactMarkdown>
                          </div>
                          {aiMeta && (
                            <div className="ai-answer-meta">
                              {aiMeta.rewrittenQuery && (
                                <div className="ai-meta-block">
                                  <span className="ai-meta-title">IR query</span>
                                  <span className="ai-meta-line">{aiMeta.rewrittenQuery}</span>
                                </div>
                              )}
                              {aiMeta.retrievalConfidence != null && (
                                <div className="ai-meta-block">
                                  <span className="ai-meta-title">Retrieval confidence</span>
                                  <span className="ai-meta-line">{aiMeta.retrievalConfidence.toFixed(3)}</span>
                                </div>
                              )}
                              {(aiMeta.results?.length ?? 0) > 0 && (
                                <div className="ai-meta-block">
                                  <span className="ai-meta-title">Top results</span>
                                  <ul className="ai-meta-list">
                                    {aiMeta.results?.slice(0, 3).map((r, i) => (
                                      <li key={`${r.player_name ?? "r"}-${i}`}>
                                        {[r.player_name, r.team].filter(Boolean).join(" · ") || "Unnamed player"}
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {(aiMeta.evidence?.length ?? 0) > 0 && (
                                <div className="ai-meta-block">
                                  <span className="ai-meta-title">Retrieved evidence</span>
                                  <ul className="ai-meta-list">
                                    {aiMeta.evidence?.slice(0, 3).map((item, i) => {
                                      const details = [
                                        item.player_name,
                                        item.team,
                                        item.season_label,
                                        typeof item.retrieval_score === "number"
                                          ? `score ${item.retrieval_score.toFixed(3)}`
                                          : null,
                                      ].filter(Boolean).join(" · ");
                                      const stats = item.key_stats
                                        ? Object.entries(item.key_stats)
                                            .filter(([, v]) => v != null)
                                            .slice(0, 3)
                                            .map(([k, v]) => `${k.replace(/_/g, " ")}: ${String(v)}`)
                                            .join(" | ")
                                        : null;
                                      return (
                                        <li key={`${item.player_name ?? "e"}-${i}`}>
                                          <div>{details}</div>
                                          {stats && <div className="ai-meta-stats">{stats}</div>}
                                        </li>
                                      );
                                    })}
                                  </ul>
                                </div>
                              )}
                              {(aiMeta.warnings?.length ?? 0) > 0 && (
                                <div className="ai-meta-block">
                                  <span className="ai-meta-title">Warnings</span>
                                  <ul className="ai-meta-list">
                                    {aiMeta.warnings?.map((w, i) => <li key={i}>{w}</li>)}
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <AnimatePresence>
                    {searchMode?.toLowerCase().includes("svd") && status === "populated" && (
                      <motion.div
                        className="svd-badge"
                        initial={{ opacity: 0, y: -8 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.26, ease: "easeOut" }}
                      >
                        <span className="svd-badge-icon">✦</span>
                        <span className="svd-badge-label">Query upgraded</span>
                        <span className="svd-badge-sep">·</span>
                        <span className="svd-badge-desc">SVD enhanced matching</span>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {svdAvailable && status === "populated" && (
                    <div className="svd-ranking-toggle">
                      <span className="svd-ranking-label">Ranking</span>
                      <button
                        type="button"
                        className={`svd-ranking-btn ${!showSvdRanking ? "active" : ""}`}
                        onClick={() => setShowSvdRanking(false)}
                      >
                        Standard
                      </button>
                      <button
                        type="button"
                        className={`svd-ranking-btn ${showSvdRanking ? "active" : ""}`}
                        onClick={() => setShowSvdRanking(true)}
                      >
                        SVD
                      </button>
                    </div>
                  )}

                  <PlayerGrid
                    players={showSvdRanking && playersSvd.length > 0 ? playersSvd : players}
                    onFullStatsClick={handleFullStatsClick}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </motion.main>

        <AnimatePresence>
          {heroMode && (
            <motion.div
              className="hero-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              onClick={() => setHeroMode(false)}
            >
              <div className="hero-overlay-inner" onClick={(e) => e.stopPropagation()}>
                <motion.div
                  className="hero-logo-wrapper"
                  initial={{ opacity: 0, scale: 0.88 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.92 }}
                  transition={{ duration: 0.28, ease: "easeOut" }}
                >
                  <Logo className="logo-hero" />
                </motion.div>

                <motion.div layoutId="main-search">
                  <SearchBar
                    value={searchTerm}
                    autoFocus
                    onChange={handleSearchChange}
                    onSubmit={() => {
                      setHeroMode(false);
                      aiMode ? void runAiSearch(searchTerm) : void runSearch(searchTerm);
                    }}
                    placeholder="look up the best Brazilian wingers..."
                    showAiToggle={useLlm}
                    aiMode={aiMode}
                    onAiToggle={() => setAiMode(m => !m)}
                  />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  transition={{ duration: 0.26, delay: 0.16, ease: "easeOut" }}
                >
                  <QueryCarousel onSelect={(q) => {
                    setSearchTerm(q);
                    setHeroMode(false);
                    void runSearch(q);
                  }} />
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <PlayerProfile player={selectedPlayer} onClose={() => setSelectedPlayer(null)} />
      </div>
    </>
  );
}

export default App;
