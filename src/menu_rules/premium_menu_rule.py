"""
Premium menu rule: max 1 premium item per day, 1-2 per week.
"""

from typing import Dict, Any, List
from ortools.sat.python import cp_model
from .base_menu_rule import (
    BaseMenuRule,
    Diagnostic,
    DiagnosticPhase,
    DiagnosticSeverity,
    DiagnoseContext,
    MenuRuleType,
)
from src.constants import BASE_SLOT_NAMES


class PremiumMenuRule(BaseMenuRule):
    """
    Config:
    {
        "type": "premium",
        "name": "premium_limits",
        "max_per_day": 1,
        "min_per_horizon": 1,
        "max_per_horizon": 2
    }
    """

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.PREMIUM
        self.max_per_day = rule_config.get('max_per_day', 1)
        self.min_per_horizon = rule_config.get('min_per_horizon', 1)
        self.max_per_horizon = rule_config.get('max_per_horizon', 2)

    def validate_config(self) -> bool:
        return not self._collect_errors()

    def validation_errors(self) -> List[str]:
        return self._collect_errors()

    def _collect_errors(self) -> List[str]:
        errs: List[str] = []
        if self.max_per_day < 0:
            errs.append(f"max_per_day must be >= 0 (got {self.max_per_day})")
        if self.min_per_horizon < 0:
            errs.append(
                f"min_per_horizon must be >= 0 (got {self.min_per_horizon})"
            )
        if self.max_per_horizon < 0:
            errs.append(
                f"max_per_horizon must be >= 0 (got {self.max_per_horizon})"
            )
        if self.min_per_horizon > self.max_per_horizon:
            errs.append(
                f"min_per_horizon ({self.min_per_horizon}) must be <= "
                f"max_per_horizon ({self.max_per_horizon})"
            )
        return errs

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        cfg = context.get('cfg')
        dates = context.get('dates', [])
        day_premium_vars = context.get('day_premium_vars', {})

        if not cfg or not cfg.premium_flag_col:
            return

        premium_day_bools = []
        for di in range(len(dates)):
            lits = day_premium_vars.get(di, [])
            prem_day = model.NewBoolVar(f'premium_day_{di}')
            if lits:
                model.Add(sum(lits) <= self.max_per_day)
                model.Add(sum(lits) == prem_day)
            else:
                model.Add(prem_day == 0)
            premium_day_bools.append(prem_day)

        total = sum(premium_day_bools)
        has_any = any(len(day_premium_vars.get(di, [])) > 0 for di in range(len(dates)))
        if has_any:
            model.Add(total >= self.min_per_horizon)
            model.Add(total <= self.max_per_horizon)
        else:
            model.Add(total == 0)

    def diagnose(self, ctx: DiagnoseContext) -> List[Diagnostic]:
        """Premium constraint requires a number of premium-flagged days
        in the horizon. Diagnose:

          - ERROR when min_per_horizon > 0 but the configured flag
            column is missing OR no items in any active slot pool have
            the flag set. The solver would silently relax in apply()
            (``if not cfg.premium_flag_col: return``), but the user's
            intent ("I want at least N premium days") is lost — surface it.
          - WARNING when premium_count < min_per_horizon (CP-SAT
            cannot promote enough premium days; multi-restart will
            eventually fail).
        """
        diags: List[Diagnostic] = []
        flag_col = ctx.cfg.premium_flag_col if ctx.cfg else None
        if not flag_col:
            if self.min_per_horizon > 0:
                diags.append(Diagnostic(
                    rule=self.name, rule_type=self.rule_type.value,
                    severity=DiagnosticSeverity.ERROR,
                    phase=DiagnosticPhase.APPLY,
                    message=(
                        f"Premium rule requires ≥{self.min_per_horizon} "
                        f"premium day(s) but no premium_flag_col is "
                        f"configured. The constraint will silently drop."
                    ),
                    suggestion=(
                        "Set SolverConfig.premium_flag_col to the column "
                        "name (e.g. 'is_premium_veg'), or set "
                        "min_per_horizon=0 in the rule config."
                    ),
                    affected={'min_per_horizon': self.min_per_horizon},
                ))
            return diags

        base_slots = ctx.active_base_slots or list(BASE_SLOT_NAMES)
        per_day_counts: Dict[str, int] = {}
        total_premium = 0
        for d in ctx.dates:
            day_count = 0
            for base in base_slots:
                if (d, base) in ctx.skip_cells:
                    continue
                pool = ctx.pools.get(base)
                if pool is None or len(pool) == 0:
                    continue
                if flag_col not in pool.columns:
                    continue
                day_count += int(pool[flag_col].fillna(0).astype(int).eq(1).sum())
            per_day_counts[d.isoformat()] = day_count
            total_premium += day_count

        if self.min_per_horizon > 0 and total_premium == 0:
            diags.append(Diagnostic(
                rule=self.name, rule_type=self.rule_type.value,
                severity=DiagnosticSeverity.ERROR,
                phase=DiagnosticPhase.APPLY,
                message=(
                    f"Premium rule requires ≥{self.min_per_horizon} "
                    f"premium day(s), but no items in any slot pool have "
                    f"{flag_col}=1."
                ),
                suggestion=(
                    f"Add at least {self.min_per_horizon} premium item"
                    f"{'s' if self.min_per_horizon != 1 else ''} to the "
                    f"ontology (set {flag_col}=1)."
                ),
                affected={
                    'min_per_horizon': self.min_per_horizon,
                    'flag_col': flag_col,
                    'total_premium': 0,
                },
            ))
        elif total_premium and self.min_per_horizon > 0:
            # Count days that *could* be premium (at least one premium
            # item in a non-skipped slot). The constraint is per-day,
            # so this is the right metric — not the raw item count.
            premium_capable_days = sum(
                1 for c in per_day_counts.values() if c > 0
            )
            if premium_capable_days < self.min_per_horizon:
                diags.append(Diagnostic(
                    rule=self.name, rule_type=self.rule_type.value,
                    severity=DiagnosticSeverity.ERROR,
                    phase=DiagnosticPhase.APPLY,
                    message=(
                        f"Premium rule needs ≥{self.min_per_horizon} "
                        f"premium day(s), but only {premium_capable_days} "
                        f"of {len(ctx.dates)} dates can carry a premium "
                        f"item (the others have 0 items with "
                        f"{flag_col}=1 in any slot pool)."
                    ),
                    suggestion=(
                        "Add premium items that match each day's theme "
                        "filters, or relax min_per_horizon."
                    ),
                    affected={
                        'min_per_horizon': self.min_per_horizon,
                        'premium_capable_days': premium_capable_days,
                        'total_dates': len(ctx.dates),
                    },
                ))
        return diags
