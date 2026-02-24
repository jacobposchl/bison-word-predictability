# ============================================================
# GLMMs for switching ~ surprisal/entropy + controls
# ============================================================

# ---- Packages ----
library(dplyr)
library(lme4)
library(sjPlot)



# ---- Load data ----

# Change from "autoregressive/" to "masked/" for masked surprisal data results. 
surprisal_results <- read.csv(
  "autoregressive/window_1/surprisal_results.csv",
  header = TRUE,
  stringsAsFactors = TRUE
)

# ---- Data prep ----
surprisal_results <- surprisal_results %>%
  mutate(
    switch_pos     = as.factor(switch_pos),
    participant_id = as.factor(participant_id)
  ) %>%
  # drop unknown POS
  filter(switch_pos != "X") %>%
  # collapse POS into noun/verb/others
  mutate(
    pos_category = case_when(
      switch_pos == "NOUN" ~ "noun",
      switch_pos == "VERB" ~ "verb",
      TRUE ~ "others"
    ),
    pos_category = factor(pos_category)
  )

# ---- Contrast coding (sum coding) ----
surprisal_results$group <- factor(
  surprisal_results$group,
  levels = c("Heritage", "Immersed", "Homeland")
)
contrasts(surprisal_results$group) <- contr.sum(3)

surprisal_results$pos_category <- factor(
  surprisal_results$pos_category,
  levels = c("noun", "verb", "others")
)
contrasts(surprisal_results$pos_category) <- contr.sum(3)

# ---- Exclude multi-word sentences + proper nouns (for later models) ----
surprisal_results_excl_multi_propn <- surprisal_results %>%
  group_by(sent_id) %>%
  filter(any(single_worded == 1)) %>%
  filter(!any(is_propn == 1 & is_switch == 1)) %>%
  ungroup()

# ---- Remove cases where switch word appears in context ----
surprisal_results_excl_multi_propn_word_in_ctx <- surprisal_results %>%
  group_by(sent_id) %>%
  filter(any(single_worded == 1)) %>%
  filter(!any(is_propn == 1 & is_switch == 1)) %>%
  filter(word_in_context == 0) %>%
  ungroup()

# ============================================================
# 1) Models on full dataset
# ============================================================

# Surprisal-only model
model_surp <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + (1 | participant_id),
  data = surprisal_results,
  family = binomial
)
summary(model_surp)
sjPlot::tab_model(model_surp, show.se = TRUE, show.stat = TRUE)

# Surprisal + entropy model
model_ent <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + entropy_context_0 + (1 | participant_id),
  data = surprisal_results,
  family = binomial
)
summary(model_ent)
sjPlot::tab_model(model_ent, show.se = TRUE, show.stat = TRUE)

# ============================================================
# 2) Exclude proper nouns (is_propn == 0)
# ============================================================

surprisal_no_propn <- surprisal_results %>%
  group_by(sent_id) %>%
  filter(!any(is_propn == 1 & is_switch == 1)) %>%
  ungroup()

model_surp_no_propn <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + (1 | participant_id),
  data = surprisal_no_propn,
  family = binomial
)
summary(model_surp_no_propn)
sjPlot::tab_model(model_surp_no_propn, show.se = TRUE, show.stat = TRUE)

model_ent_no_propn <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + entropy_context_0 + (1 | participant_id),
  data = surprisal_no_propn,
  family = binomial
)
summary(model_ent_no_propn)
sjPlot::tab_model(model_ent_no_propn, show.se = TRUE, show.stat = TRUE)

# ============================================================
# 3) Restrict to "single-worded" sentences (and exclude proper nouns)
# ============================================================

surprisal_single_no_propn <- surprisal_results %>%
  group_by(sent_id) %>%
  filter(any(single_worded == 1)) %>%
  filter(!any(is_propn == 1 & is_switch == 1)) %>%
  ungroup()

model_surp_single_no_propn <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + (1 | participant_id),
  data = surprisal_single_no_propn,
  family = binomial
)
summary(model_surp_single_no_propn)
sjPlot::tab_model(model_surp_single_no_propn, show.se = TRUE, show.stat = TRUE)

model_ent_single_no_propn <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + entropy_context_0 + (1 | participant_id),
  data = surprisal_single_no_propn,
  family = binomial
)
summary(model_ent_single_no_propn)
sjPlot::tab_model(model_ent_single_no_propn, show.se = TRUE, show.stat = TRUE)

# Compare fit: surprisal-only vs surprisal+entropy (same dataset)
anova(model_surp_single_no_propn, model_ent_single_no_propn, test = "Chisq")


# ============================================================
# 4) Interaction model: surprisal_context_0 * group
# ============================================================

model_int <- glmer(
  is_switch ~ 1+ word_length + pos_category + surprisal_context_3 * group + (1 | participant_id),
  data = surprisal_results_excl_multi_propn,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(model_int)
sjPlot::tab_model(model_int, show.se = TRUE, show.stat = TRUE)


# ============================================================
# 5) Single-worded, no proper nouns, no word-in-context
# ============================================================

model_surp_excl_ctx <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_3 * group + (1 | participant_id),
  data = surprisal_results_excl_multi_propn_word_in_ctx,
  family = binomial
)
summary(model_surp_excl_ctx)
sjPlot::tab_model(model_surp_excl_ctx, show.se = TRUE, show.stat = TRUE)