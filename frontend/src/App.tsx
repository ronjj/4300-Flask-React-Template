import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import "./App.css";
import Chat from "./Chat";
import Logo from "./components/Logo";
import SearchBar from "./components/SearchBar";
import PlayerGrid from "./components/PlayerGrid";
import { PlayerCardData, PlayerStats } from "./types";
import POPULAR_PLAYERS from "./data/popularPlayers";
import searchSvg from "./assets/search.svg";
import soccerballSvg from "./assets/soccerball.svg";
import compassSvg from "./assets/compass.svg";

const EXAMPLE_QUERIES = [
  "best brazilian wingers",
  "top scorers in La Liga",
  "fastest defenders in the Premier League",
  "creative midfielders from Argentina",
  "young strikers under 23",
  "best free kick takers",
  "most assists in Serie A",
  "tall center backs over 6ft",
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

    const speed = 0.25; // px per frame

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
      onMouseEnter={() => { pausedRef.current = true; }}
      onMouseLeave={() => { pausedRef.current = false; }}
    >
      <div className="query-carousel-track" ref={trackRef}>
        {chips.map((q, i) => (
          <button
            key={i}
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
// for stef to run deployed backend locally
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
interface SearchResponse {
  results: PlayerStats[];
}
type SearchStatus = "idle" | "loading" | "populated" | "empty" | "error";
function toCardData(results: PlayerStats[]): PlayerCardData[] {
  return results.map((player, index) => ({
    key: `${player.name}-${player.team ?? "unknown"}-${player.league ?? "unknown"}`,
    rank: index + 1,
    name: player.name,
    team: player.team,
    position: player.position,
    nationality: player.nationality,
    goals: player.goals,
    appearances: player.appearances,
    image: player.image,
  }));
}
function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [players, setPlayers] = useState<PlayerCardData[]>([]);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const searchInputRef = useRef<HTMLInputElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const loadConfig = async (): Promise<void> => {
      try {
        const response = await fetch(`${API_BASE}/api/config`);
        if (!response.ok) return;
        const data: { use_llm?: boolean } = await response.json();
        setUseLlm(Boolean(data.use_llm));
      } catch {
        // if config fails, keep useLlm = false and continue rendering UI
      }
    };
    void loadConfig();
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
    try {
      const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(trimmed)}`);
      if (!response.ok) {
        setPlayers([]);
        setStatus("error");
        return;
      }
      const data: SearchResponse = await response.json();
      const nextPlayers = toCardData(Array.isArray(data.results) ? data.results : []);
      setPlayers(nextPlayers);
      setStatus(nextPlayers.length > 0 ? "populated" : "empty");
    } catch {
      setPlayers([]);
      setStatus("error");
    }
  };
  const handleChatSearch = (term: string): void => {
    setSearchTerm(term);
    void runSearch(term);
  };

  const shellMode = useMemo<"home" | "results">(() => {
    if (players.length > 0 || status === "loading" || status === "empty" || status === "error") return "results";
    return "home";
  }, [players.length, status]);

  const focusSearch = (): void => {
    scrollToShell();
    searchInputRef.current?.focus();
  };

  const statusText =
    status === "loading"
      ? "Searching..."
      : status === "empty"
        ? "No results found."
        : status === "error"
          ? "Could not load results. Please try again."
          : null;

  return (
    <div className={`full-body-container ${useLlm ? "llm-mode" : ""}`}>
      <motion.main
        className="welcome"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
      >
        <div className="welcome-content">
          <Logo className="logo-hero" />
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
            <Logo className="logo-header" />
            <div className="header-search">
              <SearchBar
                value={searchTerm}
                inputRef={searchInputRef}
                onChange={(nextValue) => {
                  setSearchTerm(nextValue);
                  if (status !== "idle") setStatus("idle");
                }}
                onSubmit={() => void runSearch(searchTerm)}
                placeholder="look up the best Brazilian wingers..."
              />
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
                  <button type="button" className="feature-tile" onClick={focusSearch}>
                    <img src={searchSvg} alt="" aria-hidden="true" className="tile-icon" />
                    <h3 className="tile-title">Search</h3>
                    <p className="tile-subtitle">Type a player name or describe what you're looking for</p>
                  </button>
                  <button type="button" className="feature-tile" onClick={focusSearch}>
                    <img src={soccerballSvg} alt="" aria-hidden="true" className="tile-icon" />
                    <h3 className="tile-title">Discover</h3>
                    <p className="tile-subtitle">Get ranked results with key stats</p>
                  </button>
                  <button type="button" className="feature-tile" onClick={focusSearch}>
                    <img src={compassSvg} alt="" aria-hidden="true" className="tile-icon" />
                    <h3 className="tile-title">Explore</h3>
                    <p className="tile-subtitle">Dive into full player profiles</p>
                  </button>
                </div>

                <div className="popular-section">
                  <h2 className="section-title">Popular Players</h2>
                  <PlayerGrid players={POPULAR_PLAYERS} />
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
                <QueryCarousel onSelect={(q) => {
                  setSearchTerm(q);
                  void runSearch(q);
                }} />

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

                <PlayerGrid players={players} />
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </motion.main>
      {useLlm && <Chat onSearchTerm={handleChatSearch} />}
    </div>
  );
}
export default App;