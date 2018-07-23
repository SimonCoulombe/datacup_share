# data ----
final_number_of_iter <- 5  # should be 5 000  , changed to make demo faster

performance_test <- read_csv(here::here("data", "performance_test.csv"))
alltrain_dummy <- read_rds( "alltrain_dummy.rds")
test_dummy <- read_rds( "test_dummy.rds")
feature_vars_dummy <- colnames(alltrain_dummy %>% select(-Default))
alltrain_xgbmatrix <- xgb.DMatrix(
  data = alltrain_dummy  %>%  select(-Default, -contains("default")) %>% data.matrix(), 
  label = alltrain_dummy %>% pull(Default),
  missing = "NAN")


test_xgbmatrix <- xgb.DMatrix(
  data = test_dummy %>% select(-Default, -contains("default")) %>% data.matrix(), 
  label = test_dummy %>% pull(label_var),
  missing = "NAN")

# ordered AUC of options tested in 02_search_xgboost_hyperparameters 
param_results <- write_rds("param_results.rds")

################### list of param sets -----


# grab my favorite set, create a few copies (exact same or slightly modified) to reduce variance of predicted values
# for the last submission I just overwrote everything..
list_of_param_sets <- param_results  %>% head(1)  %>% select(eta, max_depth, min_child_weight, subsample, colsample_bytree, gamma, optimal_number_of_iter )
  
 list_of_param_sets_temp <- list_of_param_sets %>% 
   mutate(eta = 0.003, optimal_number_of_iter = final_number_of_iter, colsample_bytree=0.3,
          subsample = 0.5, min_child_weigth = 1, max_depth = 3, gamma =0  )
 
 list_of_param_sets_temp <- list_of_param_sets_temp %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
   bind_rows(list_of_param_sets_temp %>% mutate(optimal_number_of_iter = final_number_of_iter)) %>%
 
  mutate(rownum = row_number()) %>% nest(-rownum)

# model on full training set ----
start <- Sys.time()

solution_list <- list_of_param_sets_temp %>% mutate(booster = map(data, function(X){
  message(paste0(), " eta = ", X$eta, " max.depth = ", X$max_depth, " min_child_weigth = ", X$min_child_weight,
          " subsample = ", X$subsample, " colsample_bytree = ", X$colsample_bytree, " gamma = ", X$gamma, ", optimal_number_of_iter = ", X$optimal_number_of_iter)
  set.seed(1234)
  myParam <- list(
    booster = "gbtree",
    eta = X$eta,
    gamma = X$gamma,
    max.depth = X$max_depth,
    min_child_weight=X$min_child_weight,
    subsample = X$subsample,
    colsample_bytree = X$colsample_bytree,
    objective = 'binary:logistic' ,
    eval_metric = "auc")
  
  
  full_booster <- xgb.train(
    params = myParam, 
    data = alltrain_xgbmatrix, 
    nround = X$optimal_number_of_iter )
  
  var_importance <- xgb.importance(
    feature_names = feature_vars_dummy,
    model = full_booster) %>% tbl_df()
  

  
  test_pred <- predict(full_booster, newdata= test_xgbmatrix)
  
  function_return <- list()
  function_return$full_booster <- full_booster
  function_return$var_importance <- var_importance
  function_return$test_pred <- test_pred
  return(function_return)
}))

print(Sys.time() - start) 

# ensemble everything ----
solution <- data.frame( pred1 = solution_list$booster[[1]]$test_pred,
                         pred2 = solution_list$booster[[2]]$test_pred,                         
                        pred3 = solution_list$booster[[3]]$test_pred,
                         pred4 = solution_list$booster[[4]]$test_pred,
                         pred5 = solution_list$booster[[5]]$test_pred,
                         pred6 = solution_list$booster[[6]]$test_pred,
                         pred7 = solution_list$booster[[7]]$test_pred,
                         pred8 = solution_list$booster[[8]]$test_pred,
                        pred9 = solution_list$booster[[9]]$test_pred,
                        pred10 = solution_list$booster[[10]]$test_pred) %>%
   as_tibble() %>%
   mutate(
     final_pred = (pred1 + pred2 + pred3 + pred4 +pred5 + pred6 +pred7 + pred8 + pred9 + pred10) / 10) 

solution <- performance_test %>% select(ID_CPTE) %>% bind_cols(solution %>% select(final_pred)) %>% rename(Default  = final_pred)
saveRDS(solution, file="solution_17_2359_hailmary3.rds")
