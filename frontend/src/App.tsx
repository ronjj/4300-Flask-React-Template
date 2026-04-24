import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, LayoutGroup, motion, useScroll, useTransform } from "framer-motion";
import "./App.css";
import Chat from "./Chat";
import Logo from "./components/Logo";
import SearchBar from "./components/SearchBar";
import PlayerGrid from "./components/PlayerGrid";
import { PlayerCardData, PlayerStats, SearchResponse } from "./types";
import { PlayerCardData, PlayerStats, SvdLegendEntry } from "./types";
import POPULAR_PLAYERS from "./data/popularPlayers";
import PlayerProfile from "./components/PlayerProfile";
import searchSvg from "./assets/search.svg";
import soccerballSvg from "./assets/soccerball.svg";
import compassSvg from "./assets/compass.svg";

const EXAMPLE_QUERIES = [
  "best brazilian wingers",
  "top scorers in La Liga",
  "fastest defenders in the Premier League",
  "creative midfielders from Argentina",
  "best free kick takers",
  "most assists in Serie A",
  "clinical finishers in Bundesliga",
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

interface SearchResponse {
  results: PlayerStats[];
  svd_available?: boolean;
  results_without_svd?: PlayerStats[] | null;
  results_svd?: PlayerStats[] | null;
  svd_latent_dimensions?: SvdLegendEntry[];
}

interface SvdCompareState {
  without: PlayerCardData[];
  with: PlayerCardData[];
  legend: SvdLegendEntry[];
}

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
  const [useLlm, setUseLlm] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [players, setPlayers] = useState<PlayerCardData[]>([]);
  const [, setSvdCompare] = useState<SvdCompareState | null>(null);
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
      setStatus("idle");
      return;
    }
    scrollToShell();
    setStatus("loading");
    setSearchMode(null);
    try {
      const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(trimmed)}`);
      if (!response.ok) {
        setPlayers([]);
        setStatus("error");
        return;
      }
      const data: SearchResponse = await response.json();
      const results = Array.isArray(data.results) ? data.results : [];

      // Build rank-delta map: name → (rank_without_svd − rank_with_svd)
      // positive = SVD moved the player higher in the list
      const rankDeltaMap = new Map<string, number>();
      if (data.svd_available && data.results_without_svd && data.results_svd) {
        const withoutRanks = new Map(
          data.results_without_svd.map((p, i) => [p.name, i])
        );
        data.results_svd.forEach((player, svdIdx) => {
          const withoutIdx = withoutRanks.get(player.name);
          if (withoutIdx !== undefined) {
            rankDeltaMap.set(player.name, withoutIdx - svdIdx);
          }
        });
      }

      const nextPlayers = toCardData(results).map((card) =>
        rankDeltaMap.has(card.name)
          ? { ...card, svdRankDelta: rankDeltaMap.get(card.name)! }
          : card
      );
      setPlayers(nextPlayers);
      setStatus(nextPlayers.length > 0 ? "populated" : "empty");
      setSearchMode(data.results[0]?.search_mode ?? null);
    } catch {
      setPlayers([]);
      setStatus("error");
    }
  };

  const handleChatSearch = (term: string): void => {
    setSearchTerm(term);
    void runSearch(term);
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
    <LayoutGroup>
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
                      onSubmit={() => void runSearch(searchTerm)}
                      placeholder="look up the best Brazilian wingers..."
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

                  <PlayerGrid players={players} onFullStatsClick={handleFullStatsClick} />
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
                      void runSearch(searchTerm);
                    }}
                    placeholder="look up the best Brazilian wingers..."
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

        {useLlm && <Chat onSearchTerm={handleChatSearch} />}

        <PlayerProfile player={selectedPlayer} onClose={() => setSelectedPlayer(null)} />
      </div>
    </LayoutGroup>
  );
}

export default App;
