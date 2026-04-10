# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)
library(haven)
library(jtools)
library(xgboost)
library(devtools)
library(parallel)
library(doParallel)
library(tictoc)

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

tic() # this starts to time the OLS model so I can gather the data summarizing the number of seconds required to execute each version of the ML approach
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
og_time_ols <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as og_time_ols

tic() # this starts to time the elastic model so I can gather the data summarizing the number of seconds required to execute each version of the ML approach
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

og_time_elastic <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as og_time_elastic

tic() # this starts to time the random forest model so I can gather the data summarizing the number of seconds required to execute each version of the ML approach
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

og_time_forest <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as og_time_forest

tic() # this starts to time the eXtreme gradient boosting model so I can gather the data summarizing the number of seconds required to execute each version of the ML approach
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

og_time_extreme <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as og_time_extreme

# here I start to set up the parallelization process
local_cluster <- makeCluster(7) # using detectCores() in the console, I found that I have 8 CPU cores on my computer. I subtracted 1 to save one core for the operating system
registerDoParallel(local_cluster) # this tells caret to distribute the CV fold iterations across the local_cluster nodes

tic() # this starts the timer for the first parallelized model
ols_model_parallel <- train(
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

parallel_time_ols <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as parallel_time_ols

tic() # this starts the timer for the second parallelized model
elastic_model_parallel <- train(
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
parallel_time_elastic <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as parallel_time_elastic

tic() # this starts the timer for the third parallelized model
forest_model_parallel <- train(
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
parallel_time_forest <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as parallel_time_forest

tic() # this starts the timer for the fourth parallelized model
extreme_model_parallel <- train(
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
parallel_time_extreme <- toc() # this stops the time and will return the time elapsed
# this also stores this value with the returned time as parallel_time_extreme

stopCluster(local_cluster) # this shuts down the parallel computing cluster
registerDoSEQ() # this makes sure that caret now runs sequentially and no longer in parallel

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

table2_tbl <- tibble(
  algo = c("OLS Regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  original = c( # this finds the elapsed time for the original models
    og_time_ols$toc - og_time_ols$tic, # subtracting tic() from toc() gives the elapsed seconds
    og_time_elastic$toc - og_time_elastic$tic, # subtracting tic() from toc() gives the elapsed seconds
    og_time_forest$toc - og_time_forest$tic, # subtracting tic() from toc() gives the elapsed seconds
    og_time_extreme$toc - og_time_extreme$tic # subtracting tic() from toc() gives the elapsed seconds
  ),
  parallelized = c( # this finds the elapsed time for the parallelized models
    parallel_time_ols$toc - parallel_time_ols$tic, # subtracting tic() from toc() gives the elapsed seconds
    parallel_time_elastic$toc - parallel_time_elastic$tic, # subtracting tic() from toc() gives the elapsed seconds
    parallel_time_forest$toc - parallel_time_forest$tic, # subtracting tic() from toc() gives the elapsed seconds
    parallel_time_extreme$toc - parallel_time_extreme$tic # subtracting tic() from toc() gives the elapsed seconds
  )
)

table2_tbl
write_csv(table2_tbl, "../out/table2.csv") # this writes the csv for table 2 and puts it in the out folder
