import { useMemo, useState } from "react";
import { PlayerCardData, SvdActivationEntry, SvdAlignmentEntry } from "../types";
type PlayerCardProps = {
  data: PlayerCardData;
  onFullStatsClick?: (player: PlayerCardData) => void;
  showSvdControls?: boolean;
};
const fallbackImage =
  "https://resources.premierleague.com/premierleague25/photos/players/110x140/placeholder.png";

function SvdRow({ dim }: { dim: SvdAlignmentEntry }): JSX.Element {
  const label = dim.label ?? `Dimension ${dim.dim + 1}`;
  const dims = dim.source_dims?.length ? dim.source_dims.map((d) => d + 1).join(", ") : null;
  return (
    <div className="svd-mini-row">
      <span className="svd-mini-label">{label}</span>
      <span className={`svd-mini-value ${dim.contribution >= 0 ? "pos" : "neg"}`}>
        {dim.contribution >= 0 ? "+" : ""}{dim.contribution.toFixed(2)}
      </span>
      {dims && <span className="svd-mini-dims" title={`Aggregated dims: ${dims}`}>dims {dims}</span>}
    </div>
  );
}

function ActivationChip({ a }: { a: SvdActivationEntry }): JSX.Element {
  const label = a.label ?? `Dim ${a.dim + 1}`;
  const v = a.activation;
  const dims = a.source_dims?.length ? a.source_dims.map((d) => d + 1).join(", ") : null;
  return (
    <span
      className={`svd-chip ${v >= 0 ? "pos" : "neg"}`}
      title={dims ? `Aggregated dims: ${dims}` : undefined}
    >
      {label}: {v >= 0 ? "+" : ""}{v.toFixed(2)}
      {dims && <span className="svd-chip-dims">dims {dims}</span>}
    </span>
  );
}

function aggregateActivations(entries: SvdActivationEntry[]): SvdActivationEntry[] {
  const byLabel = new Map<
    string,
    { label: string; label_detail?: string; sum: number; abs: number; dims: number[] }
  >();
  for (const e of entries) {
    const label = e.label ?? `Dim ${e.dim + 1}`;
    const current =
      byLabel.get(label) ?? { label, label_detail: e.label_detail, sum: 0, abs: 0, dims: [] };
    current.sum += e.activation;
    current.abs += Math.abs(e.activation);
    current.dims.push(e.dim);
    // Keep first non-empty detail as a tooltip-ish context if needed later.
    if (!current.label_detail && e.label_detail) current.label_detail = e.label_detail;
    byLabel.set(label, current);
  }
  const out: SvdActivationEntry[] = Array.from(byLabel.values())
    .sort((a, b) => b.abs - a.abs)
    .map((v, i) => ({
      // dim is no longer meaningful after aggregation; keep deterministic ordering index.
      dim: i,
      label: v.label,
      label_detail: v.label_detail,
      activation: v.sum,
      source_dims: v.dims.sort((a, b) => a - b),
    }));
  return out;
}

function aggregateAlignments(entries: SvdAlignmentEntry[]): SvdAlignmentEntry[] {
  const byLabel = new Map<
    string,
    { label: string; label_detail?: string; sum: number; abs: number; dims: number[] }
  >();
  for (const e of entries) {
    const label = e.label ?? `Dimension ${e.dim + 1}`;
    const current =
      byLabel.get(label) ?? { label, label_detail: e.label_detail, sum: 0, abs: 0, dims: [] };
    current.sum += e.contribution;
    current.abs += Math.abs(e.contribution);
    current.dims.push(e.dim);
    if (!current.label_detail && e.label_detail) current.label_detail = e.label_detail;
    byLabel.set(label, current);
  }
  const out: SvdAlignmentEntry[] = Array.from(byLabel.values())
    .sort((a, b) => b.abs - a.abs)
    .map((v, i) => ({
      dim: i,
      label: v.label,
      label_detail: v.label_detail,
      // aggregated values only
      query_activation: 0,
      player_activation: 0,
      contribution: v.sum,
      top_positive_loadings: [],
      top_negative_loadings: [],
      source_dims: v.dims.sort((a, b) => a - b),
    })) as unknown as SvdAlignmentEntry[];
  return out;
}

function PlayerCard({ data, onFullStatsClick, showSvdControls = false }: PlayerCardProps): JSX.Element {
  const [showSvd, setShowSvd] = useState(false);

  const svdVectors = data.fullStats?.svd_vectors;
  const hasSvd = Boolean(
    (svdVectors?.top_alignment && svdVectors.top_alignment.length > 0) ||
      (data.fullStats?.svd_explain &&
        (data.fullStats.svd_explain.positive_dimensions.length > 0 ||
          data.fullStats.svd_explain.negative_dimensions.length > 0))
  );

  const topAlignment = useMemo(
    () => aggregateAlignments(svdVectors?.top_alignment ?? []).slice(0, 4),
    [svdVectors?.top_alignment]
  );

  const topQueryActs = useMemo(
    () => aggregateActivations(svdVectors?.query_top_activations ?? []).slice(0, 6),
    [svdVectors?.query_top_activations]
  );

  return (
    <article className="player-card">
      <img
        className="player-image"
        src={data.image || fallbackImage}
        alt={data.name}
        onError={(e) => {
          (e.currentTarget as HTMLImageElement).src = fallbackImage;
        }}
      />
      <div className="player-card-overlay" />
      <div className="player-card-content">
        <div className="player-rank-row">
          <span className="player-rank">#{data.rank}</span>
          {data.svdRankDelta !== undefined && data.svdRankDelta !== 0 && (
            <span className={`svd-rank-delta ${data.svdRankDelta > 0 ? "up" : "down"}`}>
              SVD {data.svdRankDelta > 0 ? `↑${data.svdRankDelta}` : `↓${Math.abs(data.svdRankDelta)}`}
            </span>
          )}
        </div>
        <h3 className="player-name">{data.name}</h3>
        <p className="player-team">{data.team}</p>
        <div className="player-stats">
          <div className="player-info-row">
            <span className="label">Position:</span>
            <span className="value">{data.position}</span>
          </div>
          <div className="player-info-row">
            <span className="label">Country:</span>
            <span className="value">{data.nationality}</span>
          </div>
          <div className="player-info-row">
            <span className="label">Goals:</span>
            <span className="value">{data.goals}</span>
          </div>
          <div className="player-info-row">
            <span className="label">Appearances:</span>
            <span className="value">{data.appearances}</span>
          </div>
        </div>
        <button
          type="button"
          className="full-stats-btn"
          onClick={() => onFullStatsClick?.(data)}
        >
          Full Stats
        </button>

        {hasSvd && showSvdControls && (
          <>
            <button
              type="button"
              className="show-svd-btn"
              onClick={() => setShowSvd((v) => !v)}
              aria-expanded={showSvd}
            >
              {showSvd ? "Hide SVD" : "Show SVD"}
            </button>

            {showSvd && (
              <div className="svd-mini-panel">
                {topQueryActs.length > 0 && (
                  <div className="svd-mini-section">
                    <p className="svd-mini-title">Query latent factors</p>
                    <div className="svd-chip-row">
                      {topQueryActs.map((a) => (
                        <ActivationChip key={`qa-${a.dim}`} a={a} />
                      ))}
                    </div>
                  </div>
                )}

                {topAlignment.length > 0 && (
                  <div className="svd-mini-section">
                    <p className="svd-mini-title">Top alignments</p>
                    <div className="svd-mini-list">
                      {topAlignment.map((dim) => (
                        <SvdRow key={`al-${dim.dim}`} dim={dim} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </article>
  );
}
export default PlayerCard;