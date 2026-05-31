export const RULE_LABELS: Record<string, string> = {
  RULE_DEP_NO_CLEAN: "Deposition without a clean within the prior 12 steps",
  RULE_METAL_ETCH_NO_LITHO: "Metal etch without a lithography/develop before it",
  RULE_ETCH_NO_MASK: "Patterned etch without a DEVELOP PHOTORESIST mask",
  RULE_LITHO_LEVEL_SKIP: "Lithography level applied out of order / skipped",
  RULE_IMPLANT_NO_MASK: "Implant without a patterned opening before it",
  RULE_CMP_NO_DEP: "CMP with no deposition/fill to planarize",
  RULE_PAD_OPEN_BEFORE_DEP: "Pad window opened before the metal/passivation it exposes",
  RULE_TEST_BEFORE_PASSIVATION: "Electrical test before passivation is complete",
  RULE_SHIP_BEFORE_TEST: "SHIP LOT before WAFER SORT TEST",
  RULE_BACKSIDE_BEFORE_PASSIVATION: "Backside metal before frontside passivation",
};

export function ruleLabel(rule: string): string {
  return RULE_LABELS[rule] ?? rule;
}
