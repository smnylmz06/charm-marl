from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TacticWeights:

    gegenpressing:          float = 1.0
    tiki_taka:              float = 1.0
    progression:            float = 1.0
    spacing:                float = 1.0
    counter_attack:         float = 1.0
    high_press:             float = 1.0
    zonal_marking:          float = 1.0
    offside_trap:           float = 1.0
    overlapping_fullbacks:  float = 1.0
    underlapping_runs:      float = 1.0
    possession_play:        float = 1.0
    direct_play:            float = 1.0
    wing_play:              float = 1.0
    false_9:                float = 1.0
    half_space:             float = 1.0
    overload:               float = 1.0
    switch_of_play:         float = 1.0
    third_man_runs:         float = 1.0
    buildup_play:           float = 1.0
    pressing_traps:         float = 1.0
    set_pieces:             float = 1.0
    gk_positioning:         float = 1.0
    defensive_compactness:  float = 1.0
    isolated_attacker:      float = 1.0
    transition_defense:     float = 1.0
    flank_coverage:         float = 1.0
    coordinated_press:      float = 1.0


@dataclass
class TacticConfig:
    FINAL_SCALE:            float = 0.05
    STEP_CLIP:              float = 1.0
    AFK_PENALTY:            float = -1.0
    AFK_SPEED_THRESHOLD:    float = 0.02
    MOVEMENT_BONUS:         float = 0.8
    MOVEMENT_THRESHOLD:     float = 0.02
    HOLDING_GRACE_STEPS:    int   = 1
    HOLDING_PENALTY_STEP:   float = 2.0
    SHAPE_PENALTY:          float = -0.5
    SHAPE_THRESHOLD:        float = 0.4
    PASS_STREAK_CAP:        int   = 5
    PASS_STREAK_WEIGHT:     float = 0.15

    SHORT_PASS:             int   = 9
    LONG_PASS:              int   = 10
    HIGH_PASS:              int   = 11
    SHOT_ACTION:            int   = 12
    SLIDE_TACKLE:           int   = 13
    TACKLE:                 int   = 16

    PASS_ACTIONS:           frozenset = field(default_factory=lambda: frozenset({9, 10, 11}))
    TACKLE_ACTIONS:         frozenset = field(default_factory=lambda: frozenset({13, 16}))

    PASS_BONUS_ATK:         float = 6.0
    SHOT_BONUS:             float = 5.0
    TACKLE_BONUS:           float = 8.0
    INTERCEPT_BONUS:        float = 0.6

    HOLDING_GRACE_STEPS:    int   = 3
    HOLDING_PENALTY_STEP:   float = 0.5

    GEGEN_BONUS:            float = 0.4
    GEGEN_TRANSITION_BONUS: float = 0.8
    GEGEN_DOT_THRESH:       float = 0.75
    GEGEN_SPEED_THRESH:     float = 0.015

    TIKI_SHORT_BONUS:       float = 3.0
    TIKI_TRIANGLE_BONUS:    float = 0.5
    TIKI_TRIANGLE_RADIUS:   float = 0.15

    COUNTER_BONUS:          float = 1.2
    COUNTER_SPEED_MIN:      float = 0.03
    COUNTER_FWD_MIN:        float = 0.03

    HIGH_PRESS_BONUS:       float = 0.15
    HIGH_PRESS_X_THRESH:    float = 0.10
    HIGH_PRESS_DIST_MAX:    float = 0.30

    ZONAL_BONUS:            float = 0.08
    ZONAL_TOLERANCE:        float = 0.15

    OFFSIDE_BONUS:          float = 0.5
    OFFSIDE_SYNC_TOL:       float = 0.05
    OFFSIDE_FWD_THRESH:     float = 0.01
    OFFSIDE_MIN_DEF:        int   = 3

    OVERLAP_BONUS:          float = 0.4
    OVERLAP_Y_THRESH:       float = 0.28
    OVERLAP_FWD_THRESH:     float = 0.02

    UNDERLAP_BONUS:         float = 0.35
    UNDERLAP_Y_MIN:         float = 0.12
    UNDERLAP_Y_MAX:         float = 0.25
    UNDERLAP_FWD_THRESH:    float = 0.02

    POSSESSION_BONUS:       float = 0.2

    DIRECT_BONUS:           float = 0.6
    DIRECT_SPEED_MIN:       float = 0.04
    DIRECT_FWD_MIN:         float = 0.03

    WING_BONUS:             float = 0.25
    WING_Y_THRESH:          float = 0.30

    FALSE9_BONUS:           float = 0.5
    FALSE9_X_MAX:           float = 0.30

    HALF_SPACE_BONUS:       float = 0.2
    HALF_SPACE_Y_MIN:       float = 0.12
    HALF_SPACE_Y_MAX:       float = 0.30

    OVERLOAD_BONUS:         float = 0.15
    OVERLOAD_RADIUS:        float = 0.20
    OVERLOAD_MIN_ADV:       int   = 2

    SWITCH_BONUS:           float = 0.5
    SWITCH_Y_DELTA:         float = 0.30

    THIRD_MAN_BONUS:        float = 0.30
    THIRD_MAN_FWD_THRESH:   float = 0.02

    BUILDUP_BONUS:          float = 0.40
    BUILDUP_X_MAX:          float = -0.10

    PRESS_TRAP_BONUS:       float = 0.45
    PRESS_TRAP_DOT_MIN:     float = 0.50
    PRESS_TRAP_DOT_MAX:     float = 0.85
    PRESS_TRAP_MIN_MATES:   int   = 2
    PRESS_TRAP_MATE_RADIUS: float = 0.25

    SET_PIECE_BONUS:        float = 4.0
    SET_PIECE_X_THRESH:     float = 0.60

    GK_SAFE_X_MAX:          float = -0.70
    GK_PENALTY:             float = -1.0

    COMPACT_MAX_SPREAD:     float = 0.55
    COMPACT_PENALTY:        float = -0.25

    ISOLATED_RADIUS:        float = 0.25
    ISOLATED_PENALTY:       float = -0.20

    TRANS_DEF_BONUS:        float = 0.5
    TRANS_DEF_SPEED_MIN:    float = 0.02

    FLANK_COVERAGE_BONUS:   float = 0.20
    FLANK_Y_THRESH:         float = 0.28
    FLANK_DEF_X_MAX:        float = 0.10

    COORD_PRESS_BONUS:      float = 0.35
    COORD_PRESS_DIST:       float = 0.20
    COORD_PRESS_MIN:        int   = 2

    SPACING_THRESHOLD: float = 0.15
    SPACE_BONUS: float = 0.3


class ObsContext:

    __slots__ = (
        "player_idx", "my_pos", "my_dir", "speed",
        "ball_pos", "left_team", "right_team", "left_dirs",
        "team_center", "ball_owned_team", "ball_owned_player",
        "dist_to_ball",
    )

    def __init__(self, obs: dict, role_id: int) -> None:
        self.player_idx        = role_id + 1
        self.my_pos            = np.array(obs["left_team"][self.player_idx])
        self.my_dir            = np.array(obs["left_team_direction"][self.player_idx])
        self.speed             = float(np.linalg.norm(self.my_dir))
        self.ball_pos          = np.array(obs["ball"][:2])
        self.left_team         = [np.array(p) for p in obs["left_team"]]
        self.right_team        = [np.array(p) for p in obs.get("right_team", [])]
        self.left_dirs         = [np.array(d) for d in obs["left_team_direction"]]
        self.team_center       = np.mean(np.stack(self.left_team), axis=0)
        self.ball_owned_team   = obs.get("ball_owned_team", -1)
        self.ball_owned_player = obs.get("ball_owned_player", -1)
        self.dist_to_ball      = float(np.linalg.norm(self.my_pos - self.ball_pos))

    @staticmethod
    def try_build(obs: dict, actions, role_id) -> Optional["ObsContext"]:
        role_id    = int(np.asarray(role_id).flat[0])
        player_idx = role_id + 1
        left_team  = obs.get("left_team", [])
        left_dirs  = obs.get("left_team_direction", [])
        ball       = obs.get("ball", [])
        n_actions  = int(np.asarray(actions).shape[0])

        if (role_id >= n_actions
                or len(ball) < 2
                or left_team.size == 0
                or left_dirs.size == 0
                or player_idx >= len(left_team)
                or player_idx >= len(left_dirs)):
            return None

        return ObsContext(obs, role_id)


class TacticalQualityCalculator:

    def __init__(
        self,
        cfg:     Optional[TacticConfig]  = None,
        weights: Optional[TacticWeights] = None,
    ) -> None:
        self.cfg     = cfg     or TacticConfig()
        self.weights = weights or TacticWeights()

        self.pass_streak:           int   = 0
        self.prev_ball_owned_team:  int   = -1
        self.prev_ball_y:           float = 0.0
        self.just_passed:           bool  = False
        self.ball_holding_steps:    int   = 0


    def calculate(
        self,
        obs:       dict,
        actions,
        role_id,
        role_type: str,
    ) -> float:
        role_id = int(np.asarray(role_id).flat[0])
        actions = np.asarray(actions)

        ctx = ObsContext.try_build(obs, actions, role_id)
        if ctx is None:
            return 0.0

        action = int(actions[role_id])
        score  = self._base_score(ctx, action, role_type)
        score += self._tactical_score(ctx, action, role_type)

        self._update_state(ctx, action)

        scaled = score * self.cfg.FINAL_SCALE
        return float(np.clip(scaled, -self.cfg.STEP_CLIP, self.cfg.STEP_CLIP))

    def reset(self) -> None:
        self.pass_streak          = 0
        self.prev_ball_owned_team = -1
        self.prev_ball_y          = 0.0
        self.just_passed          = False
        self.ball_holding_steps   = 0


    def _update_state(self, ctx: ObsContext, action: int) -> None:
        self.just_passed          = action in self.cfg.PASS_ACTIONS
        self.prev_ball_owned_team = ctx.ball_owned_team
        self.prev_ball_y          = float(ctx.ball_pos[1])
        self.pass_streak          = self.pass_streak + 1 if self.just_passed else 0

        if ctx.ball_owned_player == ctx.player_idx:
            self.ball_holding_steps += 1
        else:
            self.ball_holding_steps = 0


    def _base_score(self, ctx: ObsContext, action: int, role_type: str) -> float:
        score  = self._kinetic_penalty(ctx)
        score += self._team_shape_penalty(ctx)
        score += self._pass_streak_bonus()

        if   ctx.ball_owned_team == 0:
            score += self._base_attack(ctx, action, role_type)
        elif ctx.ball_owned_team == 1:
            score += self._base_defense(ctx, action, role_type)

        return score

    def _kinetic_penalty(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        if ctx.speed < cfg.AFK_SPEED_THRESHOLD:
            return cfg.AFK_PENALTY
        if ctx.speed >= cfg.MOVEMENT_THRESHOLD:
            return cfg.MOVEMENT_BONUS
        return 0.0

    def _team_shape_penalty(self, ctx: ObsContext) -> float:
        dist = np.linalg.norm(ctx.my_pos - ctx.team_center)
        return self.cfg.SHAPE_PENALTY if dist > self.cfg.SHAPE_THRESHOLD else 0.0

    def _pass_streak_bonus(self) -> float:
        return min(self.pass_streak, self.cfg.PASS_STREAK_CAP) * self.cfg.PASS_STREAK_WEIGHT

    def _base_attack(self, ctx: ObsContext, action: int, role_type: str) -> float:
        cfg   = self.cfg
        score = 0.0

        if action in cfg.PASS_ACTIONS and ctx.speed > 0.01:
            score += cfg.PASS_BONUS_ATK
        if abs(ctx.my_dir[1]) > 0.05:
            score += 0.3
        if role_type == "DF" and abs(ctx.my_pos[1]) > 0.3 and ctx.my_dir[0] > 0:
            score += 0.12

        if ctx.ball_owned_player == ctx.player_idx:
            score += self._ball_carrier(ctx, action)
        else:
            score += self._off_ball(ctx, action)

        return score

    def _ball_carrier(self, ctx: ObsContext, action: int) -> float:
        cfg   = self.cfg
        score = 0.0

        if action in cfg.PASS_ACTIONS:
            score += cfg.PASS_BONUS_ATK
            return score

        if action == cfg.SHOT_ACTION and ctx.my_pos[0] > 0.5:
            score += cfg.SHOT_BONUS
            return score

        if ctx.my_dir[0] > 0.03:
            score += 0.4
        elif ctx.speed < cfg.AFK_SPEED_THRESHOLD:
            score += cfg.AFK_PENALTY

        excess = self.ball_holding_steps - cfg.HOLDING_GRACE_STEPS
        if excess > 0:
            score -= cfg.HOLDING_PENALTY_STEP * excess

        return score

    def _off_ball(self, ctx: ObsContext, action: int = 0) -> float:
        score    = 0.0
        opp_dist = self._closest_opp_dist(ctx)

        if   opp_dist > 0.08: score += 0.20
        elif opp_dist < 0.05: score -= 0.15

        if   ctx.dist_to_ball < 0.15: score += 0.50
        elif ctx.dist_to_ball < 0.30: score += 0.25
        elif ctx.dist_to_ball > 0.60: score -= 0.30

        if ctx.my_dir[0] > 0.03: score += 0.20
        if ctx.my_dir[0] > 0.06: score += 0.20

        if self._nearby_teammate_count(ctx, radius=0.15) >= 2: score += 0.15

        if action in self.cfg.PASS_ACTIONS:
            score += 0.50

        return score

    def _base_defense(self, ctx: ObsContext, action: int, role_type: str) -> float:
        cfg   = self.cfg
        score = 0.0

        if ctx.dist_to_ball > 0:
            dir_norm = ctx.my_dir / (ctx.speed + 1e-8)
            vec_norm = (ctx.ball_pos - ctx.my_pos) / (ctx.dist_to_ball + 1e-8)
            dot      = float(np.dot(dir_norm, vec_norm))

            if   dot > 0.8 and ctx.speed > 0.01: score += cfg.INTERCEPT_BONUS
            elif 0.4 < dot <= 0.8:               score += 0.2

            if ctx.dist_to_ball < 0.08 and action in cfg.TACKLE_ACTIONS:
                score += cfg.TACKLE_BONUS

        if role_type == "DF" and ctx.my_pos[0] > 0.2: score -= 0.08
        if ctx.speed > 0.02 and ctx.my_dir[0] > 0:    score += 0.10

        return score


    def _tactical_score(self, ctx: ObsContext, action: int, role_type: str) -> float:
        w     = self.weights
        owned = ctx.ball_owned_team
        score = 0.0

        if owned == 1:
            score += w.gegenpressing        * self._gegenpressing(ctx)
            score += w.high_press           * self._high_press(ctx)
            score += w.zonal_marking        * self._zonal_marking(ctx)
            score += w.offside_trap         * self._offside_trap(ctx)
            score += w.pressing_traps       * self._pressing_traps(ctx)
            score += w.coordinated_press    * self._coordinated_press(ctx)
            score += w.flank_coverage       * self._flank_coverage(ctx, role_type)
            score += w.transition_defense   * self._transition_defense(ctx)

        score += w.gk_positioning        * self._gk_positioning(ctx, role_type)
        score += w.defensive_compactness * self._defensive_compactness(ctx, role_type)
        score += w.isolated_attacker     * self._isolated_attacker(ctx, role_type)

        if owned == 0:
            score += w.tiki_taka             * self._tiki_taka(ctx, action)
            score += w.counter_attack        * self._counter_attack(ctx)
            score += w.overlapping_fullbacks * self._overlapping_fullbacks(ctx, role_type)
            score += w.underlapping_runs     * self._underlapping_runs(ctx, role_type)
            score += w.possession_play       * self._possession_play(action)
            score += w.direct_play           * self._direct_play(ctx, action)
            score += w.wing_play             * self._wing_play(ctx)
            score += w.false_9               * self._false_9(ctx, role_type)
            score += w.half_space            * self._half_space(ctx)
            score += w.overload              * self._overload(ctx)
            score += w.switch_of_play        * self._switch_of_play(ctx, action)
            score += w.third_man_runs        * self._third_man_runs(ctx)
            score += w.buildup_play          * self._buildup_play(ctx, action, role_type)
            score += w.set_pieces            * self._set_pieces(ctx, action)

        return score


    def _gegenpressing(self, ctx: ObsContext) -> float:
        cfg   = self.cfg
        score = 0.0

        if self.prev_ball_owned_team == 0 and ctx.ball_owned_team == 1:
            score += cfg.GEGEN_TRANSITION_BONUS

        if ctx.dist_to_ball > 0:
            dir_norm = ctx.my_dir / (ctx.speed + 1e-8)
            vec_norm = (ctx.ball_pos - ctx.my_pos) / (ctx.dist_to_ball + 1e-8)
            dot      = float(np.dot(dir_norm, vec_norm))

            if dot > cfg.GEGEN_DOT_THRESH and ctx.speed > cfg.GEGEN_SPEED_THRESH:
                score += cfg.GEGEN_BONUS

        return score

    def _high_press(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        in_press_zone = ctx.my_pos[0] > cfg.HIGH_PRESS_X_THRESH
        near_ball     = ctx.dist_to_ball < cfg.HIGH_PRESS_DIST_MAX
        return cfg.HIGH_PRESS_BONUS if (in_press_zone and near_ball) else 0.0

    def _zonal_marking(self, ctx: ObsContext) -> float:
        cfg  = self.cfg
        tol  = cfg.ZONAL_TOLERANCE
        x_ok = ctx.my_pos[0] < ctx.ball_pos[0] + tol
        y_ok = abs(ctx.my_pos[1] - ctx.ball_pos[1]) < tol * 2
        return cfg.ZONAL_BONUS if (x_ok and y_ok) else 0.0

    def _offside_trap(self, ctx: ObsContext) -> float:
        cfg = self.cfg

        def_positions = [
            (i, p) for i, p in enumerate(ctx.left_team)
            if p[0] < 0.1
        ]
        if len(def_positions) < cfg.OFFSIDE_MIN_DEF:
            return 0.0

        xs = [p[0] for _, p in def_positions]
        if max(xs) - min(xs) > cfg.OFFSIDE_SYNC_TOL:
            return 0.0

        fwd_count = sum(
            1 for i, _ in def_positions
            if ctx.left_dirs[i][0] > cfg.OFFSIDE_FWD_THRESH
        )
        return cfg.OFFSIDE_BONUS if fwd_count >= cfg.OFFSIDE_MIN_DEF else 0.0

    def _pressing_traps(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        if ctx.dist_to_ball < 1e-6 or ctx.speed < 1e-6:
            return 0.0

        dir_norm = ctx.my_dir / (ctx.speed + 1e-8)
        vec_norm = (ctx.ball_pos - ctx.my_pos) / (ctx.dist_to_ball + 1e-8)
        dot      = float(np.dot(dir_norm, vec_norm))

        if not (cfg.PRESS_TRAP_DOT_MIN < dot < cfg.PRESS_TRAP_DOT_MAX):
            return 0.0

        mates_near_ball = sum(
            1 for i, mate in enumerate(ctx.left_team)
            if i != ctx.player_idx
            and np.linalg.norm(mate - ctx.ball_pos) < cfg.PRESS_TRAP_MATE_RADIUS
        )
        return cfg.PRESS_TRAP_BONUS if mates_near_ball >= cfg.PRESS_TRAP_MIN_MATES else 0.0


    def _tiki_taka(self, ctx: ObsContext, action: int) -> float:
        cfg   = self.cfg
        score = 0.0

        if action == cfg.SHORT_PASS and ctx.speed > 0.005:
            score += cfg.TIKI_SHORT_BONUS

        if self._nearby_teammate_count(ctx, radius=cfg.TIKI_TRIANGLE_RADIUS) >= 2:
            score += cfg.TIKI_TRIANGLE_BONUS

        return score

    def _counter_attack(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        transition  = (self.prev_ball_owned_team == 1 and ctx.ball_owned_team == 0)
        fast_fwd    = (ctx.speed >= cfg.COUNTER_SPEED_MIN and ctx.my_dir[0] >= cfg.COUNTER_FWD_MIN)
        return cfg.COUNTER_BONUS if (transition and fast_fwd) else 0.0

    def _overlapping_fullbacks(self, ctx: ObsContext, role_type: str) -> float:
        cfg = self.cfg
        wide_enough = abs(ctx.my_pos[1]) > cfg.OVERLAP_Y_THRESH
        moving_fwd  = ctx.my_dir[0] > cfg.OVERLAP_FWD_THRESH
        return cfg.OVERLAP_BONUS if (role_type == "DF" and wide_enough and moving_fwd) else 0.0

    def _underlapping_runs(self, ctx: ObsContext, role_type: str) -> float:
        cfg = self.cfg
        y   = abs(ctx.my_pos[1])
        in_corridor = cfg.UNDERLAP_Y_MIN < y < cfg.UNDERLAP_Y_MAX
        moving_fwd  = ctx.my_dir[0] > cfg.UNDERLAP_FWD_THRESH
        return cfg.UNDERLAP_BONUS if (role_type == "DF" and in_corridor and moving_fwd) else 0.0

    def _possession_play(self, action: int) -> float:
        return self.cfg.POSSESSION_BONUS if action in self.cfg.PASS_ACTIONS else 0.0

    def _direct_play(self, ctx: ObsContext, action: int) -> float:
        cfg        = self.cfg
        fast_fwd   = ctx.speed >= cfg.DIRECT_SPEED_MIN and ctx.my_dir[0] >= cfg.DIRECT_FWD_MIN
        long_ball  = action in (cfg.LONG_PASS, cfg.HIGH_PASS)
        return cfg.DIRECT_BONUS if (fast_fwd or long_ball) else 0.0

    def _wing_play(self, ctx: ObsContext) -> float:
        return self.cfg.WING_BONUS if abs(ctx.my_pos[1]) > self.cfg.WING_Y_THRESH else 0.0

    def _false_9(self, ctx: ObsContext, role_type: str) -> float:
        deep_position = ctx.my_pos[0] < self.cfg.FALSE9_X_MAX
        return self.cfg.FALSE9_BONUS if (role_type == "FW" and deep_position) else 0.0

    def _half_space(self, ctx: ObsContext) -> float:
        cfg           = self.cfg
        y             = abs(ctx.my_pos[1])
        in_half_space = cfg.HALF_SPACE_Y_MIN < y < cfg.HALF_SPACE_Y_MAX
        moving_fwd    = ctx.my_dir[0] > 0.01
        return cfg.HALF_SPACE_BONUS if (in_half_space and moving_fwd) else 0.0

    def _overload(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        r   = cfg.OVERLOAD_RADIUS

        our_count = sum(1 for p in ctx.left_team  if np.linalg.norm(p - ctx.ball_pos) < r)
        opp_count = sum(1 for p in ctx.right_team if np.linalg.norm(p - ctx.ball_pos) < r)

        advantage = our_count - opp_count
        if advantage <= 0:
            return 0.0
        return cfg.OVERLOAD_BONUS * min(advantage, cfg.OVERLOAD_MIN_ADV)

    def _switch_of_play(self, ctx: ObsContext, action: int) -> float:
        cfg = self.cfg
        if action not in cfg.PASS_ACTIONS:
            return 0.0
        y_delta = abs(ctx.my_pos[1] - self.prev_ball_y)
        return cfg.SWITCH_BONUS if y_delta > cfg.SWITCH_Y_DELTA else 0.0

    def _third_man_runs(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        is_runner = (
            self.just_passed
            and ctx.ball_owned_player != ctx.player_idx
            and ctx.my_dir[0] > cfg.THIRD_MAN_FWD_THRESH
        )
        return cfg.THIRD_MAN_BONUS if is_runner else 0.0

    def _buildup_play(self, ctx: ObsContext, action: int, role_type: str) -> float:
        cfg     = self.cfg
        in_own  = ctx.my_pos[0] < cfg.BUILDUP_X_MAX
        is_back = role_type in ("DF", "MF")
        is_pass = action in cfg.PASS_ACTIONS
        return cfg.BUILDUP_BONUS if (in_own and is_back and is_pass) else 0.0

    def _set_pieces(self, ctx: ObsContext, action: int) -> float:
        cfg = self.cfg
        in_box = ctx.my_pos[0] > cfg.SET_PIECE_X_THRESH
        is_shot = action == cfg.SHOT_ACTION
        return cfg.SET_PIECE_BONUS if (in_box and is_shot) else 0.0


    def _gk_positioning(self, ctx: ObsContext, role_type: str) -> float:
        if role_type != "GK":
            return 0.0
        score = 0.0
        if ctx.my_pos[0] > self.cfg.GK_SAFE_X_MAX:
            score += self.cfg.GK_PENALTY
        if ctx.speed < self.cfg.AFK_SPEED_THRESHOLD:
            score += self.cfg.AFK_PENALTY * 0.5
        return score

    def _defensive_compactness(self, ctx: ObsContext, role_type: str) -> float:
        if role_type not in ("DF", "MF", "GK"):
            return 0.0
        cfg = self.cfg
        def_positions = [
            p for i, p in enumerate(ctx.left_team)
            if i != ctx.player_idx
        ]
        if len(def_positions) < 2:
            return 0.0
        xs = [p[0] for p in def_positions]
        ys = [p[1] for p in def_positions]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        return cfg.COMPACT_PENALTY if spread > cfg.COMPACT_MAX_SPREAD else 0.0

    def _isolated_attacker(self, ctx: ObsContext, role_type: str) -> float:
        if role_type != "FW":
            return 0.0
        has_support = any(
            i != ctx.player_idx
            and np.linalg.norm(ctx.my_pos - mate) < self.cfg.ISOLATED_RADIUS
            for i, mate in enumerate(ctx.left_team)
        )
        return 0.0 if has_support else self.cfg.ISOLATED_PENALTY

    def _transition_defense(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        just_lost = (self.prev_ball_owned_team == 0 and ctx.ball_owned_team == 1)
        if not just_lost:
            return 0.0
        running_back = ctx.my_dir[0] < -cfg.TRANS_DEF_SPEED_MIN
        fast_enough  = ctx.speed > cfg.TRANS_DEF_SPEED_MIN
        return cfg.TRANS_DEF_BONUS if (running_back and fast_enough) else 0.0

    def _flank_coverage(self, ctx: ObsContext, role_type: str) -> float:
        cfg = self.cfg
        if role_type not in ("DF",):
            return 0.0
        wide      = abs(ctx.my_pos[1]) > cfg.FLANK_Y_THRESH
        defensive = ctx.my_pos[0] < cfg.FLANK_DEF_X_MAX
        return cfg.FLANK_COVERAGE_BONUS if (wide and defensive) else 0.0

    def _coordinated_press(self, ctx: ObsContext) -> float:
        cfg = self.cfg
        if not ctx.right_team:
            return 0.0
        ball_carrier_pos = ctx.ball_pos

        mates_pressing = sum(
            1 for i, mate in enumerate(ctx.left_team)
            if i != ctx.player_idx
            and np.linalg.norm(mate - ball_carrier_pos) < cfg.COORD_PRESS_DIST
        )
        self_pressing = ctx.dist_to_ball < cfg.COORD_PRESS_DIST
        if self_pressing and mates_pressing >= cfg.COORD_PRESS_MIN - 1:
            return cfg.COORD_PRESS_BONUS
        return 0.0


    def _closest_opp_dist(self, ctx: ObsContext) -> float:
        if not ctx.right_team:
            return float("inf")
        return float(min(np.linalg.norm(ctx.my_pos - opp) for opp in ctx.right_team))

    def _nearby_teammate_count(self, ctx: ObsContext, radius: float = 0.10) -> int:
        return sum(
            1 for i, mate in enumerate(ctx.left_team)
            if i != ctx.player_idx and np.linalg.norm(ctx.my_pos - mate) < radius
        )

    def _calculate_open_space_reward(self, ctx: ObsContext) -> float:
        nearby_mates = sum(1 for p in ctx.left_team
                        if np.linalg.norm(ctx.my_pos - p) < self.cfg.SPACING_THRESHOLD)

        closest_opp = min([np.linalg.norm(ctx.my_pos - opp) for opp in ctx.right_team])

        reward = 0.0
        if nearby_mates > 1:
            reward -= self.cfg.SPACING_BONUS * (nearby_mates - 1)
        if closest_opp > 0.12:
            reward += self.cfg.SPACE_BONUS

        return reward
