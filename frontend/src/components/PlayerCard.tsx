import { PlayerCardData } from "../types";
type PlayerCardProps = {
  data: PlayerCardData;
  onFullStatsClick?: (player: PlayerCardData) => void;
};
const fallbackImage =
  "https://resources.premierleague.com/premierleague25/photos/players/110x140/placeholder.png";
function PlayerCard({ data, onFullStatsClick }: PlayerCardProps): JSX.Element {
  return (
    <article className="player-card">
      <p className="player-rank">#{data.rank}</p>
      <div className="player-top">
        <img
          className="player-image"
          src={data.image || fallbackImage}
          alt={data.name}
          onError={(e) => {
            (e.currentTarget as HTMLImageElement).src = fallbackImage;
          }}
        />
        <div className="player-identity">
          <h3 className="player-name">{data.name}</h3>
          <p className="player-team">{data.team}</p>
        </div>
      </div>
      <div className="player-info-row">
        <span className="label">Position</span>
        <span className="value">{data.position}</span>
      </div>
      <div className="player-info-row">
        <span className="label">Country</span>
        <span className="value">{data.nationality}</span>
      </div>
      <div className="player-info-row">
        <span className="label">Goals</span>
        <span className="value">{data.goals}</span>
      </div>
      <div className="player-info-row">
        <span className="label">Appearances</span>
        <span className="value">{data.appearances}</span>
      </div>
      {data.similarity_score != null && (
        <div className="player-info-row">
          <span className="label">Similarity</span>
          <span className="value">{data.similarity_score.toFixed(3)}</span>
        </div>
      )}
      {data.svd_explain != null && (
        <details className="svd-explain">
          <summary>SVD explainability (latent dimensions)</summary>
          <p className="svd-explain-note">
            Each term is q<sub>d</sub>×p<sub>d</sub> before cosine normalization. Same sign on dimension d →
            positive product (reinforces the match); opposite signs → negative product.
          </p>
          <div className="svd-explain-cols">
            <div>
              <strong className="svd-explain-heading">Strongest positive q×p</strong>
              <ul className="svd-explain-list">
                {data.svd_explain.positive_dimensions.map((d) => (
                  <li key={`p-${d.dim}`}>
                    <span className="svd-dim">
                      Dim {d.dim}
                      {d.label ? ` — ${d.label}` : ""}
                    </span>
                    : q={d.query_activation.toFixed(2)}, p=
                    {d.player_activation.toFixed(2)}, q×p={d.contribution.toFixed(3)}
                    <div className="svd-loadings">
                      {d.label_detail ? (
                        <div className="svd-dim-detail">{d.label_detail}</div>
                      ) : null}
                      +V<sup>T</sup>: {d.top_positive_loadings.join(", ")} · −V<sup>T</sup>:{" "}
                      {d.top_negative_loadings.join(", ")}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <strong className="svd-explain-heading">Strongest negative q×p</strong>
              <ul className="svd-explain-list">
                {data.svd_explain.negative_dimensions.map((d) => (
                  <li key={`n-${d.dim}`}>
                    <span className="svd-dim">
                      Dim {d.dim}
                      {d.label ? ` — ${d.label}` : ""}
                    </span>
                    : q={d.query_activation.toFixed(2)}, p=
                    {d.player_activation.toFixed(2)}, q×p={d.contribution.toFixed(3)}
                    <div className="svd-loadings">
                      {d.label_detail ? (
                        <div className="svd-dim-detail">{d.label_detail}</div>
                      ) : null}
                      +V<sup>T</sup>: {d.top_positive_loadings.join(", ")} · −V<sup>T</sup>:{" "}
                      {d.top_negative_loadings.join(", ")}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </details>
      )}
      <button
        type="button"
        className="full-stats-btn"
        onClick={() => onFullStatsClick?.(data)}
      >
        Full Stats
      </button>
    </article>
  );
}
export default PlayerCard;