# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(jtools)
library(xgboost)
library(devtools)

# Data Import and Cleaning
gss_tbl <- read_sav("../data/GSS2016.sav", user_na = TRUE) %>% 
  zap_missing() %>% 
  mutate(mosthrs = as.numeric(mosthrs)) %>% 
  filter(!is.na(mosthrs)) %>%
  select(-hrs1, -hrs2) %>%
  select_if(~ mean(is.na(.)) < 0.75)

# Visualization
ggplot(gss_tbl, aes(x = mosthrs)) +
  geom_histogram(binwidth = 10, fill = "#A80000") +
  xlab("Work Hours") +
  ylab("Count") +
  theme_apa()

# Analysis
set.seed(42)
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = 0.25, list = F)
gss_holdout <- gss_tbl[holdout_indices, ]
gss_training <- gss_tbl[-holdout_indices, ]

fold_indices <- createFolds(gss_training$mosthrs, k = 10)
gss_training <- gss_training %>% 
  mutate(mosthrs = as.numeric(mosthrs))

ols_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "lm",
  na.action = na.pass,
  preProcess = "medianImpute",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

elastic_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "glmnet",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale"),
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  ),
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)
  )
)

forest_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "ranger",
  na.action = na.pass,
  preProcess = "medianImpute",
  tuneLength = 10,
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

extreme_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = "medianImpute",
  tuneLength = 10,
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

# Publication
holdout_rsq <- function(model, holdout) {
  predicted <- predict(model, holdout, na.action = na.pass)
  cor(predicted, holdout$mosthrs)^2
}

table1_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(
    max(ols_model$results$Rsquared, na.rm = TRUE),
    max(elastic_model$results$Rsquared, na.rm = TRUE),
    max(forest_model$results$Rsquared, na.rm = TRUE),
    max(extreme_model$results$Rsquared, na.rm = TRUE)
  ),
  ho_rsq = c(
    holdout_rsq(ols_model, gss_holdout),
    holdout_rsq(elastic_model, gss_holdout),
    holdout_rsq(forest_model, gss_holdout),
    holdout_rsq(extreme_model, gss_holdout)
  )
) %>% 
  mutate(across(c(cv_rsq, ho_rsq), ~ formatC(round(.x, 2), format = "f", digits = 2) %>% 
                  str_remove("^0")))

table1_tbl
write_csv(table1_tbl, "../out/table1.csv")
