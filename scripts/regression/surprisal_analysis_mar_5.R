# ============================================================
# GLMMs for switching ~ surprisal/entropy + controls
# ============================================================

# ---- Packages ----
library(dplyr)
library(lme4)
library(sjPlot)
library(emmeans)



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
    participant_id = as.factor(participant_id),
    switch_id = as.factor(sent_id)
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

# ============================================================
# 3) Restrict to "single-worded" sentences (and exclude proper nouns)
# ============================================================

surprisal_single_no_propn <- surprisal_results %>%
  group_by(sent_id) %>%
  filter(any(single_worded == 1)) %>%
  filter(!any(is_propn == 1 & is_switch == 1)) %>%
  ungroup()


# surprisal model
model_surp_single_no_propn_c0 <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_0 + (1 | participant_id),
  data = surprisal_single_no_propn,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(model_surp_single_no_propn_c0)
sjPlot::tab_model(model_surp_single_no_propn_c0, show.se = TRUE, show.stat = TRUE, transform = NULL,
  pred.labels = c("Intercept", "Word Length",
                  "Group: Heritage", "Group: Immersed",
                  "POS: Noun", "POS: Verb",
                  "Surprisal (context 0)"))


model_surp_single_no_propn_c3 <- glmer(
  is_switch ~ word_length + group + pos_category + surprisal_context_3 + (1 | participant_id),
  data = surprisal_single_no_propn,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(model_surp_single_no_propn_c3)
sjPlot::tab_model(model_surp_single_no_propn_c3, show.se = TRUE, show.stat = TRUE, transform = NULL,
  pred.labels = c("Intercept", "Word Length",
                  "Group: Heritage", "Group: Immersed",
                  "POS: Noun", "POS: Verb",
                  "Surprisal (context 3)"))


# ============================================================
# 4) Interaction model: surprisal * group
# ============================================================

# surprisal model
model_surp_int_c0 <- glmer(
  is_switch ~ 1+ word_length + pos_category + surprisal_context_0 * group + (1 | participant_id),
  data = surprisal_results_excl_multi_propn,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(model_surp_int_c0)
sjPlot::tab_model(model_surp_int_c0, show.se = TRUE, show.stat = TRUE, transform = NULL,
  pred.labels = c("Intercept", "Word Length",
                  "POS: Noun", "POS: Verb",
                  "Surprisal (context 0)",
                  "Group: Heritage", "Group: Immersed",
                  "Surprisal (context 0) x Heritage",
                  "Surprisal (context 0) x Immersed"))

model_surp_int_c3 <- glmer(
  is_switch ~ 1+ word_length + pos_category + surprisal_context_3 * group + (1 | participant_id),
  data = surprisal_results_excl_multi_propn,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)
summary(model_surp_int_c3)
sjPlot::tab_model(model_surp_int_c3, show.se = TRUE, show.stat = TRUE, transform = NULL,
  pred.labels = c("Intercept", "Word Length",
                  "POS: Noun", "POS: Verb",
                  "Surprisal (context 3)",
                  "Group: Heritage", "Group: Immersed",
                  "Surprisal (context 3) x Heritage",
                  "Surprisal (context 3) x Immersed"))


# ============================================================
# All-levels significance: each group/POS vs. grand mean
# ============================================================
# Estimates are Odds Ratios (OR) vs. grand mean. OR > 1 = more switching; OR < 1 = less.
# P-values are unadjusted (matching the sjPlot table output).

# --- model_control ---
summary(contrast(emmeans(model_control, ~ group), "eff", adjust = "none"), type = "response")
summary(contrast(emmeans(model_control, ~ pos_category), "eff", adjust = "none"), type = "response")

# --- model_surp_single_no_propn_c0 ---
summary(contrast(emmeans(model_surp_single_no_propn_c0, ~ group), "eff", adjust = "none"), type = "response")
summary(contrast(emmeans(model_surp_single_no_propn_c0, ~ pos_category), "eff", adjust = "none"), type = "response")

# --- model_surp_single_no_propn_c3 ---
summary(contrast(emmeans(model_surp_single_no_propn_c3, ~ group), "eff", adjust = "none"), type = "response")
summary(contrast(emmeans(model_surp_single_no_propn_c3, ~ pos_category), "eff", adjust = "none"), type = "response")

# --- model_surp_int_c0: does surprisal effect differ by group? (all 3 levels) ---
# Each row: does this group's surprisal slope deviate from the average slope?
# Matches the sjPlot interaction terms, but now shows Homeland too.
tmp_c0 <- as.data.frame(contrast(emtrends(model_surp_int_c0, ~ group, var = "surprisal_context_0"), "eff", adjust = "none"))
tmp_c0$OR <- exp(tmp_c0$estimate)
tmp_c0[, c("contrast", "OR", "SE", "z.ratio", "p.value")]

# --- model_surp_int_c3: does surprisal effect differ by group? (all 3 levels) ---
tmp_c3 <- as.data.frame(contrast(emtrends(model_surp_int_c3, ~ group, var = "surprisal_context_3"), "eff", adjust = "none"))
tmp_c3$OR <- exp(tmp_c3$estimate)
tmp_c3[, c("contrast", "OR", "SE", "z.ratio", "p.value")]

