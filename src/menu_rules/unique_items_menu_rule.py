"""
Unique items menu rule: each item at most once per planning session.

Uses item_to_vars from context (built by solver) to enforce uniqueness.
"""

import logging
from typing import Dict, Any

from ortools.sat.python import cp_model
from .base_menu_rule import BaseMenuRule, MenuRuleType
from src.constants import REPEATABLE_ITEM_BASES

logger = logging.getLogger(__name__)


class UniqueItemsMenuRule(BaseMenuRule):
    """
    Config:
    {
        "type": "unique_items",
        "name": "unique_items_session",
        "scope": "session"
    }
    """

    def __init__(self, rule_config: Dict[str, Any]):
        super().__init__(rule_config)
        self.rule_type = MenuRuleType.UNIQUE_ITEMS
        self.scope = rule_config.get('scope', 'session').lower()

    def validate_config(self) -> bool:
        return self.scope in ('session',)

    def apply(self, model: cp_model.CpModel, variables: Dict[str, Any],
              menu_data: Any, context: Dict[str, Any]) -> None:
        item_to_vars = context.get('item_to_vars', {})
        if not item_to_vars:
            return
        repeatable = set(REPEATABLE_ITEM_BASES)
        for item_base, vars_ in item_to_vars.items():
            if item_base not in repeatable:
                model.Add(sum(vars_) <= 1)
