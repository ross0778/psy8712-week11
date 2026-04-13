# Script Settings and Resources
library(dplyr)
library(stringr)
library(readr)
library(ranger)
library(caret)
library(haven)
library(xgboost)
library(parallel)
library(doParallel)
library(tictoc)

# Data Import and Cleaning
gss_tbl <- read_sav("../data/GSS2016.sav", user_na = TRUE) %>% 
  zap_missing() %>% 
  mutate(across(where(haven::is.labelled), as.numeric)) %>% 
  mutate(mosthrs = as.numeric(mosthrs)) %>% 
  filter(!is.na(mosthrs)) %>%
  select(-hrs1, -hrs2) %>%
  select(-id, -year, -fileversion, -ballot, cohort, -dateintv, -conrinc, -coninc, -ballotformwt, -ballotformwtnr) %>% # added to try and get rid of extra variables to troubleshoot for MSI
  select_if(~ mean(is.na(.)) < 0.75)

# Analysis
set.seed(42)
holdout_indices <- createDataPartition(gss_tbl$mosthrs, p = 0.25, list = F)
gss_holdout <- gss_tbl[holdout_indices, ]
gss_training <- gss_tbl[-holdout_indices, ]

fold_indices <- createFolds(gss_training$mosthrs, k = 10)
gss_training <- gss_training %>% 
  mutate(mosthrs = as.numeric(mosthrs))

tic()
ols_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "lm",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)
og_time_ols <- toc()

tic()
forest_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  tuneLength = 3, # shorter for time
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

og_time_forest <- toc()

tic()
extreme_model <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  tuneLength = 3, # shorter for time
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

og_time_extreme <- toc()

number_cores <- detectCores() # this detects the cores on the node
local_cluster <- makeCluster(number_cores) # this is modified so that all cores detected are now used instead of 7
registerDoParallel(local_cluster)

tic()
ols_model_parallel <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "lm",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)

parallel_time_ols <- toc()


tic()
forest_model_parallel <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "ranger",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  tuneLength = 3, # shorter for time
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)
parallel_time_forest <- toc()

tic()
extreme_model_parallel <- train(
  mosthrs ~ .,
  data = gss_training,
  method = "xgbLinear",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "nzv", "scale"), # update to troubleshoot for MSI
  tuneLength = 3, # shorter for time
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = TRUE,
    indexOut = fold_indices
  )
)
parallel_time_extreme <- toc()

stopCluster(local_cluster)
registerDoSEQ()

# Publication
holdout_rsq <- function(model, holdout) {
  predicted <- predict(model, holdout, na.action = na.pass)
  cor(predicted, holdout$mosthrs)^2
}

table3_tbl <- tibble( # changed to be table 3
  algo = c("OLS Regression", "Random Forest", "eXtreme Gradient Boosting"),
  cv_rsq = c(
    max(ols_model$results$Rsquared, na.rm = TRUE),
    max(forest_model$results$Rsquared, na.rm = TRUE),
    max(extreme_model$results$Rsquared, na.rm = TRUE)
  ),
  ho_rsq = c(
    holdout_rsq(ols_model, gss_holdout),
    holdout_rsq(forest_model, gss_holdout),
    holdout_rsq(extreme_model, gss_holdout)
  )
) %>% 
  mutate(across(c(cv_rsq, ho_rsq), ~ formatC(round(.x, 2), format = "f", digits = 2) %>% 
                  str_remove("^0")))

table3_tbl # changed to be table 3
write_csv(table3_tbl, "../out/table3.csv") # changed to be table 3

table4_tbl <- tibble( # changed to be table 4
  algo = c("OLS Regression", "Random Forest", "eXtreme Gradient Boosting"),
  supercomputer = c( # this changes the first column name to supercomputer
    og_time_ols$toc - og_time_ols$tic,
    og_time_forest$toc - og_time_forest$tic,
    og_time_extreme$toc - og_time_extreme$tic
  )
)
table4_tbl[[paste0("supercomputer_", number_cores)]] <- c( # because I wanted the number of cores in the column name to be dynamic, I had to break this step creating the column names into two steps. This line dynamically adds the number of cores in the column name
    parallel_time_ols$toc - parallel_time_ols$tic,
    parallel_time_forest$toc - parallel_time_forest$tic,
    parallel_time_extreme$toc - parallel_time_extreme$tic
  )

table4_tbl # changed to be table 4
write_csv(table4_tbl, "../out/table4.csv") # changed to be table 4

# Which models benefited most from moving to the supercomputer and why?
# I was not able to successfully get the batch script to run so I am not sure, but I would guess that the models that take are more computationally heavy would benefit the most.

# What is the relationship between time and the number of cores used?
# Again, I was not able to compare since my batch script would not run, but I would imagine that for the computationally heavy models such as random forest and xgbLinear, the time would decrease as the number of cores increases. 

# If your supervisor asked you to pick a model for use in a production model, would you recommend using the supercomputer and why? Consider all four tables when providing an answer.
# Although I do not have my results, I would pick the model that explains the most variance. Because we're using the supercomputer, I would imagine that the time it takes to run the model becomes far less important than it would on my own laptop, so going with the model that explains the most variance would be the best choice.


