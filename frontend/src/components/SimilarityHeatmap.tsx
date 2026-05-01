import { motion } from 'framer-motion'

interface Props {
  features: string[]
  players: { name: string; scores: number[]; raw: number[] }[]
}

function shortLabel(feat: string): string {
  const map: Record<string, string> = {
    goals: 'Goals',
    assists: 'Assists',
    shots_on_target: 'Shots',
    progressive_passes: 'Prog Pass',
    key_passes: 'Key Pass',
    dribbles_completed: 'Dribbles',
    tackles: 'Tackles',
    interceptions: 'Intercept',
    appearances: 'Apps',
  }
  return map[feat] ?? feat.replace(/_/g, ' ')
}

function cellColor(score: number): string {
  // Low → dark teal, High → bright green. Blends through the brand palette.
  const r = Math.round(16 + score * (115 - 16))
  const g = Math.round(185 + score * (220 - 185))
  const b = Math.round(116 + score * (80 - 116))
  const a = 0.12 + score * 0.72
  return `rgba(${r},${g},${b},${a})`
}

export default function SimilarityHeatmap({ features, players }: Props): JSX.Element {
  const cols = features.length

  return (
    <motion.div
      className="similarity-heatmap"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.26, ease: 'easeOut' }}
    >
      <div className="heatmap-header">
        <span className="heatmap-title">Similarity Heatmap</span>
        <span className="heatmap-subtitle">feature contributions across top results</span>
      </div>

      <div
        className="heatmap-grid"
        style={{ gridTemplateColumns: `minmax(100px,160px) repeat(${cols}, minmax(52px,1fr))` }}
      >
        {/* Column headers */}
        <div className="heatmap-corner" />
        {features.map(f => (
          <div key={f} className="heatmap-col-header" title={f.replace(/_/g, ' ')}>
            {shortLabel(f)}
          </div>
        ))}

        {/* Player rows */}
        {players.map(({ name, scores, raw }) => (
          <div key={name} className="heatmap-row-contents">
            <div className="heatmap-row-label" title={name}>
              {name.split(' ').slice(-1)[0]}
            </div>
            {scores.map((score, ci) => (
              <div
                key={ci}
                className="heatmap-cell"
                style={{ background: cellColor(score) }}
                title={`${name} · ${features[ci].replace(/_/g, ' ')}: ${raw[ci]}`}
              >
                <span className="heatmap-cell-val">{raw[ci]}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="heatmap-legend">
        <span className="heatmap-legend-label">Low</span>
        <div className="heatmap-legend-bar" />
        <span className="heatmap-legend-label">High</span>
      </div>
    </motion.div>
  )
}
