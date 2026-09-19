# =============================================================================
# Clasificación de deterioro cognitivo (Healthy / GDS3 / GDS4) con un árbol de
# decisión — tidymodels + rpart
#
# Autora: Valery Quintal Méndez
# Asignatura: Aprendizaje Automático Supervisado (curso 2025-26)
#
# Uso: abrir R en la RAÍZ del repositorio y ejecutar
#        source("R/arbol_decision.R")
# Entrada : data/data_set.sav
# Salida  : figures/*.png
# =============================================================================

# ── PAQUETES ─────────────────────────────────────────────────────────────────
library(tidymodels)
library(dplyr)
library(foreign)      # read.spss()
library(rpart)
library(rpart.plot)
library(rsample)
library(ggplot2)
library(scales)
library(forcats)
library(ranger)       # motor de Random Forest
library(kknn)         # motor de k-NN

dir.create("figures", showWarnings = FALSE)

# ══════════════════════════════════════════════════════════════════════════════
#  1. LECTURA Y EXPLORACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
data <- read.spss("data/data_set.sav", to.data.frame = TRUE)
glimpse(data)
summary(data)

# Tabla descriptiva: % de pacientes con el dominio alterado, por grupo
descriptive <- data %>%
  group_by(group) %>%
  summarise(across(verbal_memory:visuo,
                   ~ paste0(sum(. == "impaired", na.rm = TRUE) / n() * 100, "%"))) %>%
  t() %>%
  as.data.frame()

descriptive

# ══════════════════════════════════════════════════════════════════════════════
#  2. DIVISIÓN DE DATOS 80/20 (estratificada por group)
# ══════════════════════════════════════════════════════════════════════════════
set.seed(123)
data_split    <- initial_split(data, prop = 0.8, strata = group)
data_training <- training(data_split)
data_test     <- testing(data_split)

summary(data_training$group)   # 40 / 40 / 40
summary(data_test$group)       # 10 / 10 / 10

# ══════════════════════════════════════════════════════════════════════════════
#  3. MODELO (hiperparámetros a optimizar)
# ══════════════════════════════════════════════════════════════════════════════
dt_model <- decision_tree(
  cost_complexity = tune(),
  tree_depth      = tune(),
  min_n           = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# ══════════════════════════════════════════════════════════════════════════════
#  4. RECETA DE PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
data_recipe <- recipe(group ~ ., data = data_training) %>%
  step_dummy(all_nominal_predictors())   # p. ej. verbal_memory_impaired (0/1)

# ══════════════════════════════════════════════════════════════════════════════
#  5. WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════
data_wkfl <- workflow() %>%
  add_model(dt_model) %>%
  add_recipe(data_recipe)

# ══════════════════════════════════════════════════════════════════════════════
#  6. VALIDACIÓN CRUZADA 10-fold (estratificada)
# ══════════════════════════════════════════════════════════════════════════════
set.seed(123)
data_folds <- vfold_cv(data_training, v = 10, strata = group)

# ══════════════════════════════════════════════════════════════════════════════
#  7. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
data_metrics <- metric_set(accuracy, sens, spec, kap)

# ══════════════════════════════════════════════════════════════════════════════
#  8. AJUSTE DE HIPERPARÁMETROS (grid aleatorio de 20 combinaciones)
# ══════════════════════════════════════════════════════════════════════════════
set.seed(123)
data_grid <- grid_random(
  extract_parameter_set_dials(dt_model),
  size = 20
)
data_grid

data_tuning <- data_wkfl %>%
  tune_grid(
    resamples = data_folds,
    grid      = data_grid,
    metrics   = data_metrics
  )

# Resultados completos y top 5 por accuracy
data_tuning %>% collect_metrics()
data_tuning %>% show_best(metric = "accuracy", n = 5)

# Mejor modelo y workflow final
best_model <- data_tuning %>% select_best(metric = "accuracy")
best_model

final_wkfl <- data_wkfl %>% finalize_workflow(best_model)

# ══════════════════════════════════════════════════════════════════════════════
#  9. ENTRENAMIENTO FINAL Y EVALUACIÓN EN TEST
# ══════════════════════════════════════════════════════════════════════════════
data_final_fit <- final_wkfl %>%
  last_fit(split = data_split, metrics = data_metrics)

data_final_fit %>% collect_metrics()

compare <- as.data.frame(data_final_fit$.predictions)

# ══════════════════════════════════════════════════════════════════════════════
#  10. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════

## Árbol de decisión ----------------------------------------------------------
tree_fit <- data_final_fit %>% extract_fit_parsnip()

png("figures/arbol_decision.png", width = 2400, height = 1600, res = 200)
rpart.plot(
  tree_fit$fit,
  type          = 5,
  extra         = 104,
  roundint      = FALSE,
  fallen.leaves = TRUE,
  box.palette   = list("#e84545", "#3d5a80", "#2e8b57"),
  shadow.col    = "gray70",
  main          = "Árbol de Decisión — Clasificación Deterioro Cognitivo",
  cex           = 0.85
)
dev.off()

## Matriz de confusión --------------------------------------------------------
conf_data <- compare %>%
  count(.pred_class, group) %>%
  group_by(group) %>%
  mutate(
    pct   = n / sum(n),
    label = paste0(n, "\n(", percent(pct, accuracy = 1), ")")
  ) %>%
  ungroup() %>%
  mutate(
    .pred_class = factor(.pred_class, levels = c("healthy", "GDS3", "GDS4")),
    group       = factor(group,       levels = c("healthy", "GDS3", "GDS4"))
  )

n_test <- nrow(compare)

p_cm <- ggplot(conf_data, aes(x = group, y = fct_rev(.pred_class), fill = pct)) +
  geom_tile(color = "white", linewidth = 1.8) +
  geom_text(aes(label = label), size = 5, fontface = "bold", color = "white") +
  scale_fill_gradientn(
    colours = c("#d6e4f0", "#2e75b6", "#1f4e79"),
    values  = c(0, 0.5, 1),
    labels  = percent_format(),
    name    = "% sobre clase real"
  ) +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(
    title    = "Matriz de Confusión",
    subtitle = paste0("Árbol de Decisión — Conjunto de test (n = ", n_test, ")"),
    x        = "Clase Real (Truth)",
    y        = "Clase Predicha (Prediction)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title      = element_text(face = "bold", size = 16, hjust = 0.5, color = "#1f4e79"),
    plot.subtitle   = element_text(color = "grey45", hjust = 0.5, size = 11),
    axis.text       = element_text(face = "bold", size = 12),
    axis.title      = element_text(size = 11, color = "grey35"),
    panel.grid      = element_blank(),
    legend.position = "right",
    plot.margin     = margin(10, 10, 10, 10)
  )

ggsave("figures/confusion_matrix.png", p_cm, width = 7, height = 5.5, dpi = 300)

## Exploración de hiperparámetros ---------------------------------------------
hp_results <- data_tuning %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  arrange(desc(mean))

p_hp <- ggplot(hp_results, aes(x = cost_complexity, y = mean,
                               color = factor(tree_depth), size = min_n)) +
  geom_point(alpha = 0.8) +
  scale_x_log10(labels = scientific_format()) +
  scale_color_brewer(palette = "Blues", name = "tree_depth") +
  scale_size_continuous(range = c(3, 8), name = "min_n") +
  labs(
    title    = "Exploración de Hiperparámetros",
    subtitle = "Accuracy media en validación cruzada (10-fold)",
    x        = "cost_complexity (escala log)",
    y        = "Accuracy media (CV)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", color = "#1f4e79", hjust = 0.5),
    plot.subtitle = element_text(color = "grey45", hjust = 0.5)
  )

ggsave("figures/hiperparametros.png", p_hp, width = 8, height = 5, dpi = 300)

# ══════════════════════════════════════════════════════════════════════════════
#  11. COMPARACIÓN DE MODELOS (mismas folds, misma receta)
# ══════════════════════════════════════════════════════════════════════════════

# Árbol con los hiperparámetros óptimos + competidores con valores por defecto
modelo_dt <- decision_tree(cost_complexity = best_model$cost_complexity,
                           tree_depth      = best_model$tree_depth,
                           min_n           = best_model$min_n) %>%
  set_engine("rpart") %>% set_mode("classification")

modelo_rf  <- rand_forest(trees = 500) %>%
  set_engine("ranger") %>% set_mode("classification")

modelo_knn <- nearest_neighbor(neighbors = 5) %>%
  set_engine("kknn") %>% set_mode("classification")

wf_dt  <- workflow() %>% add_model(modelo_dt)  %>% add_recipe(data_recipe)
wf_rf  <- workflow() %>% add_model(modelo_rf)  %>% add_recipe(data_recipe)
wf_knn <- workflow() %>% add_model(modelo_knn) %>% add_recipe(data_recipe)

# Folds compartidos (misma semilla que en el ajuste de hiperparámetros)
set.seed(123)
folds <- vfold_cv(data_training, v = 10, strata = group)

eval_modelo <- function(wf, nombre) {
  fit_rs <- fit_resamples(wf, resamples = folds, metrics = data_metrics,
                          control = control_resamples(save_pred = TRUE))
  collect_metrics(fit_rs) %>% mutate(modelo = nombre)
}

resultados <- bind_rows(
  eval_modelo(wf_dt,  "Decision Tree"),
  eval_modelo(wf_rf,  "Random Forest"),
  eval_modelo(wf_knn, "k-NN")
)

resultados %>%
  select(modelo, .metric, mean, std_err) %>%
  arrange(.metric, desc(mean))

plot_data <- resultados %>%
  filter(.metric %in% c("accuracy", "sens", "spec", "kap")) %>%
  mutate(
    .metric = recode(.metric,
                     accuracy = "Accuracy",
                     sens     = "Sensitivity",
                     spec     = "Specificity",
                     kap      = "Kappa"),
    modelo = fct_reorder(modelo, mean, .fun = max)
  )

colores_metrica <- c(
  "Accuracy"    = "#E05C5C",
  "Sensitivity" = "#C4A827",
  "Specificity" = "#2E8B57",
  "Kappa"       = "#3A7FBF"
)

p_comp <- ggplot(plot_data, aes(x = mean, y = modelo, color = .metric)) +
  geom_errorbarh(aes(xmin = mean - std_err, xmax = mean + std_err),
                 height = 0.25, linewidth = 0.7) +
  geom_point(size = 3.5) +
  facet_wrap(~ .metric, scales = "free_x", ncol = 2) +
  scale_color_manual(values = colores_metrica, name = ".metric") +
  scale_x_continuous(labels = number_format(accuracy = 0.01)) +
  labs(
    title   = "Model Metrics Comparison",
    x       = "Score",
    y       = "Model",
    caption = "Media ± error estándar en validación cruzada (10-fold)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", size = 16, hjust = 0.5, color = "#1F4E79"),
    strip.text         = element_text(face = "bold", size = 12),
    axis.text.y        = element_text(size = 11),
    axis.text.x        = element_text(size = 9),
    panel.grid.major.y = element_line(color = "grey88", linewidth = 0.4),
    panel.grid.major.x = element_line(color = "grey88", linewidth = 0.4),
    panel.grid.minor   = element_blank(),
    legend.position    = "right",
    plot.caption       = element_text(color = "grey50", size = 8.5, hjust = 0.5),
    plot.margin        = margin(15, 15, 10, 15)
  )

ggsave("figures/comparacion_modelos.png", p_comp, width = 12, height = 7, dpi = 300)

# FIN
