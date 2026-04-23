import { motion } from "framer-motion";
import PlayerCard from "./PlayerCard";
import { PlayerCardData } from "../types";

type PlayerGridProps = {
  players: PlayerCardData[];
  onFullStatsClick?: (player: PlayerCardData) => void;
};

const container = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.09,
      delayChildren: 0.04,
    },
  },
};

const item = {
  hidden: { opacity: 0, y: 36 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1] as const },
  },
};

function PlayerGrid({ players, onFullStatsClick }: PlayerGridProps): JSX.Element {
  if (players.length === 0) {
    return <></>;
  }

  return (
    <motion.section
      className="player-grid"
      aria-label="Search results"
      key={players.map((p) => p.key).join(",")}
      variants={container}
      initial="hidden"
      animate="show"
    >
      {players.map((player) => (
        <motion.div key={player.key} variants={item} className="player-card-motion">
          <PlayerCard
            data={player}
            onFullStatsClick={onFullStatsClick}
          />
        </motion.div>
      ))}
    </motion.section>
  );
}

export default PlayerGrid;
