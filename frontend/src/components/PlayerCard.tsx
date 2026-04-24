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
      </div>
    </article>
  );
}
export default PlayerCard;