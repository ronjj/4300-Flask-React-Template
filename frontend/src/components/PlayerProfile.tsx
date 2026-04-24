import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PlayerCardData, SvdDimensionExplain, SvdExplain } from "../types";
import RadarChart from "./RadarChart";

type Props = {
  player: PlayerCardData | null;
  onClose: () => void;
};

const fallbackImage =
  "https://resources.premierleague.com/premierleague25/photos/players/110x140/placeholder.png";

function StatCard({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="profile-stat-card">
      <span className="profile-stat-value">{value ?? "—"}</span>
      <span className="profile-stat-label">{label}</span>
    </div>
  );
}

function ExtraRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="profile-extra-row">
      <span className="profile-extra-label">{label}</span>
      <span className="profile-extra-value">{value ?? "—"}</span>
    </div>
  );
}

function SvdDimRow({
  dim,
  maxAbs,
  positive,
}: {
  dim: SvdDimensionExplain;
  maxAbs: number;
  positive: boolean;
}): JSX.Element {
  const widthPct = (Math.abs(dim.contribution) / maxAbs) * 100;
  const label = dim.label ?? `Dimension ${dim.dim + 1}`;
  const features = positive ? dim.top_positive_loadings : dim.top_negative_loadings;

  return (
    <div className="svd-dim-item">
      <div className="svd-dim-header">
        <span className={`svd-dim-arrow ${positive ? "pos" : "neg"}`}>
          {positive ? "↑" : "↓"}
        </span>
        <span className="svd-dim-label">{label}</span>
        <span className={`svd-dim-value ${positive ? "pos" : "neg"}`}>
          {positive ? "+" : ""}{dim.contribution.toFixed(2)}
        </span>
      </div>
      <div className="svd-dim-bar-track">
        <div
          className={`svd-dim-bar-fill ${positive ? "pos" : "neg"}`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
      {features.length > 0 && (
        <p className="svd-dim-features">{features.slice(0, 3).join(" · ")}</p>
      )}
    </div>
  );
}

function SvdBreakdown({ explain }: { explain: SvdExplain }): JSX.Element {
  const posDims = explain.positive_dimensions.slice(0, 3);
  const negDims = explain.negative_dimensions.slice(0, 2);
  const allDims = [...posDims, ...negDims];
  if (allDims.length === 0) return <></>;

  const maxAbs = Math.max(...allDims.map((d) => Math.abs(d.contribution)), 0.001);

  return (
    <div className="svd-breakdown">
      {posDims.map((dim) => (
        <SvdDimRow key={`pos-${dim.dim}`} dim={dim} maxAbs={maxAbs} positive />
      ))}
      {negDims.length > 0 && (
        <>
          <div className="svd-breakdown-divider" />
          {negDims.map((dim) => (
            <SvdDimRow key={`neg-${dim.dim}`} dim={dim} maxAbs={maxAbs} positive={false} />
          ))}
        </>
      )}
    </div>
  );
}

function PlayerProfile({ player, onClose }: Props): JSX.Element {
  useEffect(() => {
    document.body.style.overflow = player ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [player]);

  const fs = player?.fullStats;

  const goalsPerGame =
    fs?.goals_per_game ??
    (player?.goals != null && player?.appearances ? player.goals / player.appearances : null);
  const assistsPerGame = fs?.assists_per_game ?? null;
  const shotAccuracy = fs?.shot_on_target_ratio ?? null;
  const dribblesPerGame =
    fs?.dribbles_completed != null && fs?.appearances
      ? fs.dribbles_completed / fs.appearances
      : null;
  const consistency = player?.appearances ? Math.min(player.appearances / 400, 1) : null;

  const radarAxes = [
    { label: "Finishing",    value: Math.min((goalsPerGame   ?? 0) / 1.2, 1) },
    { label: "Creativity",   value: Math.min((assistsPerGame ?? 0) / 0.8, 1) },
    { label: "Accuracy",     value: shotAccuracy ?? 0 },
    { label: "Dribbling",    value: Math.min((dribblesPerGame ?? 0) / 4,   1) },
    { label: "Consistency",  value: consistency ?? 0 },
  ];

  return (
    <AnimatePresence>
      {player && (
        <motion.div
          className="profile-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.22 }}
          onClick={onClose}
        >
          <motion.div
            className="profile-panel"
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="profile-hero">
              <img
                className="profile-hero-img"
                src={player.image || fallbackImage}
                alt={player.name}
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).src = fallbackImage;
                }}
              />
              <div className="profile-hero-gradient" />
              <button className="profile-close-btn" onClick={onClose} aria-label="Close">✕</button>
              <div className="profile-hero-text">
                <h2 className="profile-name">{player.name}</h2>
                <p className="profile-meta">
                  {[player.team, player.position, player.nationality].filter(Boolean).join(" · ")}
                </p>
              </div>
            </div>

            <div className="profile-body">

              <div className="profile-stats-row">
                <StatCard label="Goals"       value={fs?.goals       ?? player.goals} />
                <StatCard label="Assists"     value={fs?.assists} />
                <StatCard label="Appearances" value={fs?.appearances ?? player.appearances} />
                <StatCard label="Minutes"     value={fs?.minutes != null ? fs.minutes.toLocaleString() : undefined} />
              </div>

              <div className="profile-radar-section">
                <p className="profile-section-label">Player Profile</p>
                <div className="profile-radar-wrapper">
                  <RadarChart axes={radarAxes} size={300} />
                </div>
              </div>

              {fs && (
                <div className="profile-detail-section">
                  <p className="profile-section-label">Detailed Stats</p>
                  <div className="profile-detail-grid">
                    <ExtraRow label="Goals per Game"    value={fs.goals_per_game?.toFixed(2)} />
                    <ExtraRow label="Assists per Game"  value={fs.assists_per_game?.toFixed(2)} />
                    <ExtraRow label="Shots on Target"   value={fs.shots_on_target} />
                    <ExtraRow
                      label="Shot Accuracy"
                      value={fs.shot_on_target_ratio != null
                        ? `${(fs.shot_on_target_ratio * 100).toFixed(0)}%`
                        : undefined}
                    />
                    <ExtraRow label="Dribbles Completed" value={fs.dribbles_completed} />
                    <ExtraRow
                      label="League"
                      value={fs.league}
                    />
                  </div>
                </div>
              )}

              {fs?.svd_explain && (
                <div className="profile-svd-section">
                  <p className="profile-section-label">Match Breakdown</p>
                  <SvdBreakdown explain={fs.svd_explain} />
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default PlayerProfile;
