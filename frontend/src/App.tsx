import { useEffect, useState, useRef } from "react";
import "./App.css";
import Chat from "./Chat";
import Logo from "./components/Logo";
import SearchBar from "./components/SearchBar";
import PlayerGrid from "./components/PlayerGrid";
import { PlayerCardData, PlayerStats, SearchResponse, SvdLegendEntry } from "./types";

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
// for stef to run deployed backend locally
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

type SearchStatus = "idle" | "loading" | "populated" | "empty" | "error";

type SvdCompareState = {
  without: PlayerCardData[];
  with: PlayerCardData[];
  legend: SvdLegendEntry[];
};

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
    similarity_score: player.similarity_score ?? undefined,
    svd_explain: player.svd_explain,
  }));
}
function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [players, setPlayers] = useState<PlayerCardData[]>([]);
  const [svdCompare, setSvdCompare] = useState<SvdCompareState | null>(null);
  const [status, setStatus] = useState<SearchStatus>("idle");
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
  const runSearch = async (term: string): Promise<void> => {
    const trimmed = term.trim();
    if (trimmed === "") {
      setPlayers([]);
      setSvdCompare(null);
      setStatus("idle");
      return;
    }
    setStatus("loading");
    try {
      const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(trimmed)}`);
      if (!response.ok) {
        setPlayers([]);
        setSvdCompare(null);
        setStatus("error");
        return;
      }
      const data: SearchResponse = await response.json();
      const results = Array.isArray(data.results) ? data.results : [];
      const nextPlayers = toCardData(results);
      setPlayers(nextPlayers);
      if (
        data.svd_available &&
        data.results_without_svd != null &&
        data.results_svd != null
      ) {
        setSvdCompare({
          without: toCardData(data.results_without_svd),
          with: toCardData(data.results_svd),
          legend: data.svd_latent_dimensions ?? [],
        });
      } else {
        setSvdCompare(null);
      }
      setStatus(nextPlayers.length > 0 ? "populated" : "empty");
    } catch {
      setPlayers([]);
      setSvdCompare(null);
      setStatus("error");
    }
  };
  const handleChatSearch = (term: string): void => {
    setSearchTerm(term);
    void runSearch(term);
  };
  return (
    <div className={`full-body-container ${useLlm ? "llm-mode" : ""}`}>
      <div className="top-text">
        <Logo />
        <SearchBar
          value={searchTerm}
          onChange={(nextValue) => {
            setSearchTerm(nextValue);
            if (status !== "idle") setStatus("idle");
          }}
          onSubmit={() => void runSearch(searchTerm)}
          placeholder="look up the best Brazilian wingers..."
        />
        <QueryCarousel onSelect={(q) => {
          setSearchTerm(q);
          void runSearch(q);
        }} />
      </div>
      {status === "loading" && <p className="search-feedback">Searching...</p>}
      {status === "empty" && <p className="search-feedback">No results found.</p>}
      {status === "error" && (
        <p className="search-feedback">Could not load results. Please try again.</p>
      )}
      {svdCompare != null ? (
        <div className="svd-compare-wrap">
          <p className="svd-compare-note">
            <strong>SVD comparison.</strong> Left: rankings by cosine similarity in the original scaled stat
            vectors. Right: same query prototype projected with TruncatedSVD, then cosine similarity in latent
            space. Expand a card on the right for per-dimension q×p explainability (positive vs negative latent
            alignment).
          </p>
          <div className="svd-compare-grid">
            <div className="svd-compare-col">
              <h2>Without SVD (original feature space)</h2>
              <PlayerGrid players={svdCompare.without} />
            </div>
            <div className="svd-compare-col">
              <h2>With SVD (latent space)</h2>
              <PlayerGrid players={svdCompare.with} />
            </div>
          </div>
          {svdCompare.legend.length > 0 && (
            <details className="svd-legend">
              <summary>Latent dimensions — interpretation from Vᵀ loadings</summary>
              <ul className="svd-legend-list">
                {svdCompare.legend.map((e) => (
                  <li key={e.dim}>
                    <strong>Dim {e.dim}</strong>
                    {e.label ? ` — ${e.label}` : ""}
                    {e.explained_variance_ratio != null &&
                      ` (variance explained ${(e.explained_variance_ratio * 100).toFixed(2)}%)`}
                    {e.label_detail ? ` — ${e.label_detail}` : ""}
                    {!e.label_detail &&
                      `: strongest +weights on ${e.top_positive_loadings.join(", ")}; strongest −weights on ${e.top_negative_loadings.join(", ")}.`}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ) : (
        <PlayerGrid players={players} />
      )}
      {useLlm && <Chat onSearchTerm={handleChatSearch} />}
    </div>
  );
}
export default App;