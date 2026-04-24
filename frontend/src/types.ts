export interface SvdDimensionExplain {
  dim: number
  label?: string
  label_detail?: string
  query_activation: number
  player_activation: number
  contribution: number
  top_positive_loadings: string[]
  top_negative_loadings: string[]
}

export interface SvdExplain {
  positive_dimensions: SvdDimensionExplain[]
  negative_dimensions: SvdDimensionExplain[]
}

export interface SvdLegendEntry {
  dim: number
  label?: string
  label_detail?: string
  top_positive_loadings: string[]
  top_negative_loadings: string[]
  explained_variance_ratio?: number
}

export interface PlayerStats {
  player_id?: string | null
  name: string
  nationality: string | null
  position: string | null
  league: string
  team: string | null
  image: string | null
  goals: number | null
  assists: number | null
  appearances: number | null
  minutes: number | null
  shots_on_target: number | null
  dribbles_completed: number | null
  season_years: number[]
  seasons: string[]
  goals_per_game: number | null
  assists_per_game: number | null
  shot_on_target_ratio: number | null
  similarity_score?: number | null
  search_mode?: string | null
  svd_explain?: SvdExplain
}

export interface SearchResponse {
  mode?: string
  results: PlayerStats[]
  results_svd?: PlayerStats[] | null
  results_without_svd?: PlayerStats[] | null
  svd_available?: boolean
  svd_latent_dimensions?: SvdLegendEntry[]
}

export interface PlayerChatResult {
  player_id?: string | null
  player_name?: string | null
  position?: string | null
  league?: string | null
  team?: string | null
  final_player_score?: number | null
  best_row_score?: number | null
  avg_top2_row_score?: number | null
  player_profile_score?: number | null
}

export interface PlayerChatEvidence {
  evidence_id?: string
  source_type?: string
  player_id?: string | null
  player_name?: string | null
  season_id?: string | null
  season_label?: string | null
  team?: string | null
  league?: string | null
  position?: string | null
  retrieval_mode?: string
  retrieval_score?: number | null
  rank?: number | null
  matched_filters?: Record<string, unknown>
  style_matches?: Array<{ term?: string; stat_family?: string }>
  key_stats?: Record<string, unknown>
  provenance?: {
    dataset?: string
    row_id?: string
    retrieved_at?: string
  }
}

export interface PlayerChatResponse {
  answer?: string
  results?: PlayerChatResult[]
  evidence?: PlayerChatEvidence[]
  retrieval_confidence?: number | null
  warnings?: string[]
  error?: string
}

export interface PlayerCardData {
  key: string
  rank: number
  name: string
  team: string | null
  position: string | null
  nationality: string | null
  goals: number | null
  appearances: number | null
  image: string | null
  fullStats?: PlayerStats
}
