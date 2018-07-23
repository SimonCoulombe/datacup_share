# charger toutes les librairies de 01_important_wrangle.R.

max_nrounds <- 5e0 # était 5000 mais on veut que ça roule pour le demo..
combien_de_modeles <- 6  # était 200 dans la vraie vie  , mais on veut que ça roule pour le démo


# functions ----

get_xgboost_fold_pred_model <- function(
  # La foncton get_xgboost_fold_pred_model remplace un peu caret car elle permet de calculer des out of fold predictions et de sauvegarder le modèle final. 
  # son avantage c'est qu'elle permet de passet des offset, ce qui est utile dans les modèles de poisson .  Inutile ici, mais j'ai  mes habitudes.
  params = list(),
  data,   # categorical variables have been replaced by dummies.
  label_var,  
  feature_vars,
  offset_var = NULL,
  nb_fold = 10,
  nround = 1000,
  contraintes= NULL) {
  
  result_get_xgboost_fold_pred_model <- list()
  myfolds <- caret::createFolds(
    data %>% pull(label_var), 
    k = nb_fold, list = FALSE)
  
  # inspired by  Codes/Fonctions/get_expected_lift.R
  map_result <- seq_len(nb_fold) %>% purrr::map(~{
    function_result  <- list()
    message(paste0("calcul du fold ", .x))
    train_fold_xgbmatrix <- xgb.DMatrix(
      data = data[myfolds != .x,] %>% select(feature_vars) %>% data.matrix(), 
      label = data[myfolds != .x,] %>% pull(label_var),
      missing = "NAN")
    
    test_fold_xgbmatrix <- xgb.DMatrix(
      data = data[myfolds == .x,] %>% select(feature_vars) %>% data.matrix(), 
      label = data[myfolds == .x,] %>% pull(label_var),
      missing = "NAN")
    
    if(!is.null(offset_var)){
      setinfo(train_fold_xgbmatrix,"base_margin", data[myfolds != .x,] %>% 
                pull(offset_var) %>% log() )
      setinfo(test_fold_xgbmatrix,"base_margin", data[myfolds == .x,] %>% 
                pull(offset_var) %>% log() )}
    
    if(!is.null(contraintes)){
      booster <- xgb.train(
        params = params, 
        data = train_fold_xgbmatrix, 
        nround = nround,
        monotone_constraints= contraintes$sens)
    }
    else {
      booster <- xgb.train(
        params = params, 
        data = train_fold_xgbmatrix, 
        nround = nround)      
    }
    
    
    function_result$pred <-predict(booster, newdata= test_fold_xgbmatrix) %>% as_tibble() %>%
      bind_cols( data %>% mutate(rownum = row_number())%>%.[myfolds ==.x,] %>%
                   select(rownum)) %>%
      mutate(fold = .x)
    
    function_result$model <- booster
    
    
    return(function_result)
  }) 
  
  
  
  result_get_xgboost_fold_pred_model$pred <-  map_df(map_result, "pred") %>%
    arrange(rownum) %>%
    pull(value)
  
  result_get_xgboost_fold_pred_model$model <- map(map_result, "model")
  
  return(result_get_xgboost_fold_pred_model)
}  


get_optimal_number_of_iteration <-   function(
  params = list(),
  data,   # categorical variables have been replace with dummies
  label_var = "GPERTS",
  feature_vars, 
  offset_var = "pred_saison",
  contraintes = NULL,
  nb_fold = 10, 
  seed = 8484, 
  metric_name = "eval_poisson_nloglik",
  max_iter = 2e3,
  maximize= FALSE)
{
  # La fonction get_optimal_number_of_iteration prend des paramètres xgboost et l'applique à nfold en suivant une watchlist. 
  # lorsque ça arrête de s'améliorer, on arrête.  On retourne la valeur médiane du nombre d'arbre optimal pour chaque fold.
  # crédit: Guillaume Lepage
  
  set.seed(seed)
  
  myfolds <- caret::createFolds(
    data %>% pull(label_var), 
    k = nb_fold, list = FALSE)
  
  result_fold <- seq_len(nb_fold) %>% purrr::map_df(~{
    message(paste0("calcul du fold ", .x))
    train_fold_xgbmatrix <- xgb.DMatrix(
      data = data[myfolds != .x,] %>% select(feature_vars) %>% data.matrix(), 
      label = data[myfolds != .x,] %>% pull(label_var),
      missing = "NAN")
    
    test_fold_xgbmatrix <- xgb.DMatrix(
      data = data[myfolds == .x,] %>% select(feature_vars) %>% data.matrix(), 
      label = data[myfolds == .x,] %>% pull(label_var),
      missing = "NAN")
    
    if(!is.null(offset_var)){
      setinfo(train_fold_xgbmatrix,"base_margin", data[myfolds != .x,] %>% 
                pull(offset_var) %>% log() )
      setinfo(test_fold_xgbmatrix,"base_margin", data[myfolds == .x,] %>% 
                pull(offset_var) %>% log() )}
    
    myWatch <- list(wlist = test_fold_xgbmatrix, train = train_fold_xgbmatrix)
    ## todo : comment faire pour avoir des contraintes propres au lieu de if else?
    if(!is.null(contraintes)){
      booster <- xgb.train(
        params = params, 
        data = train_fold_xgbmatrix, 
        nround = max_iter,
        watchlist,
        verbose=0,
        callbacks = list(cb.early.stop(metric_name = metric_name, stopping_rounds = 50, maximize=maximize)),
        monotone_constraints= contraintes$sens)
    }
    else{
      booster <- xgb.train(
        params = params, 
        data = train_fold_xgbmatrix, 
        nround = max_iter,
        watchlist= myWatch,
        verbose=0,
        callbacks = list(cb.early.stop(metric_name = metric_name, stopping_rounds = 50, maximize=maximize)))
      }
    
    message("Best iter fold # ", .x, "\t",  booster$best_iteration, "\t",  Sys.time())
    data_frame(Fold = .x, best_iter = booster$best_iteration, best_score = booster$best_score)
  })
  result_fold %>% summarise(m = median(best_iter)) %>% as.numeric %>% floor
}


# param set ----

#Ici on détermine  l'ensemble des hyperparamètres qui seront utilisés.  Avant je faisais des grid.
# ici, j'ai fait une recherche aléatoire (à la bergstra)
# pendant le concours j'ai appris l'existence de jakob-r/mlrHyperopt , ce sera ma nouvelle façon.

#  Si je voulais importer les prédictions du réseau neuronal ou du glmnet pour un stack ce serait ici, mais ça ne sert à rien.
alltrain_dummy <- read_rds(path="alltrain_dummy.rds")

# Importantion du NN et du GLMNET et du lightGBM , pas utilisé finalement car le stacking / ensemble ne donne pas de gains
# pred_nn_test <-read_csv("pred_nn_test.csv")
# pred_nn_train <-read_csv("pred_nn_train.csv") %>% select(preds) %>%
#   rename(pred_nn = preds)
# 
# pred_glmnet_train <- read_csv("pred_glmnet_train.csv") %>%
#   rename(pred_glmnet = oui)
# pred_glmnet_test <- read_csv("pred_glmnet_test.csv") %>% select(oui) %>%
#   rename(pred_glmnet = oui)
# alltrain_dummy <- mydb_dummy  %>% filter(!is.na(Default))  %>%
#   bind_cols(pred_nn_train) %>% bind_cols(pred_glmnet_train)
# test_dummy <- mydb_dummy %>% filter(is.na(Default))%>%
#   bind_cols(pred_nn_test) %>% bind_cols(pred_glmnet_test)
# model_lightgbm <- read_rds("mylightgbm800.rds")


# Grid -- not used , remplacé par random search
# max_depth <-data.frame(max_depth = c(3,4,3,4))
# colsample_bytree <- data.frame(colsample_bytree =c( 0.5,0.5,.4,.4))
# subsample <- data.frame(subsample= c(  0.5,0.5, 0.5,0.5))
# min_child_weight <- data.frame(min_child_weight = c(3,1,1,3))
# eta <- data.frame(eta =c(0.01,0.01,0.01,0.01))
# gamma <-  data.frame(gamma = c(0,0,1,1))
# nrounds <- data.frame(nrounds = c(1200,1300,1400,1100))


# Random search - used

 
max_depth <-data.frame(max_depth = floor(runif(combien_de_modeles)*5 )+3)
colsample_bytree <- data.frame(colsample_bytree =runif(combien_de_modeles)*0.8 +0.2)
subsample <- data.frame(subsample =runif(combien_de_modeles)*0.8 +0.2)
min_child_weight <- data.frame(min_child_weight = floor(runif(combien_de_modeles)*10 )+1)
eta <- data.frame(eta = runif(combien_de_modeles)*0.06 +0.002)
gamma <-  data.frame(gamma =c(rep(0,combien_de_modeles/2), runif(combien_de_modeles/2)*10))
nrounds <- data.frame(nrounds = rep(max_nrounds,combien_de_modeles)) #

df.params <- max_depth %>%
  bind_cols(colsample_bytree ) %>%
  bind_cols(subsample) %>%
  bind_cols(min_child_weight) %>%
  bind_cols(eta) %>%
  bind_cols(gamma) %>%
  bind_cols(nrounds) %>%
  as_tibble() %>%
  mutate(rownum = row_number(),
  rownumber = row_number())

 feature_vars <- colnames(alltrain_dummy %>% select(-Default))
 label_var <- "Default"
 
#write_rds(df.params, path = "df.params7.rds")

list_of_param_sets <- df.params %>% nest(-rownum)

# model ----

start <- Sys.time()

z <- list_of_param_sets %>% mutate(booster = map(data, function(X){
  message(paste0("model #", X$rownumber), " eta = ", X$eta, " max.depth = ", X$max_depth, " min_child_weigth = ", X$min_child_weight,
          " subsample = ", X$subsample, " colsample_bytree = ", X$colsample_bytree, ", gamma = ",  X$gamma," , nrounds = ", X$nrounds)
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
  
## find the optimal number of tree
  optimal_number_of_iter <- get_optimal_number_of_iteration(
    params = myParam,
    data= alltrain_dummy  %>% select(one_of(c(label_var, feature_vars))),
    label_var = label_var,
    offset_var = NULL,
    feature_vars= feature_vars,
    nb_fold = 9, # était 9 pour de vrai
    max_iter = X$nrounds,
    contraintes = NULL,
    metric_name = "wlist_auc",
    maximize= TRUE)
   
  optimal_number_of_iter <- X$nrounds
  #  train model with the optimal number of trees
  out_of_fold_pred_model <- get_xgboost_fold_pred_model(
    params = myParam,
    data= alltrain_dummy  %>% select(one_of(c(label_var, feature_vars))),
    label_var = label_var,  
    offset_var = NULL,
    feature_vars= feature_vars,
    nb_fold = 9, #était 9 pour de vrai
    nround = optimal_number_of_iter, # X$nrounds
    contraintes = NULL
  )
  
  zz <- alltrain_dummy %>% select(Default) %>% bind_cols(out_of_fold_pred_model$pred %>% as.tibble)
  roc_obj <- roc(zz$Default, zz$value)
  auc <- auc(roc_obj) %>% as.tibble()
  
  function_return <- list()
  function_return$auc <- auc
  function_return$optimal_number_of_iter <- optimal_number_of_iter
  function_return$out_of_fold_pred_model <- out_of_fold_pred_model
  message(paste0("auc = ",auc))
  return(function_return)
}))

print(Sys.time() - start) # Time difference of 1.423422 mins pour 2 modeles avec 3 fols.. mettons 1minute 30 par modèle avec 5 folds


auc_iter <- data.frame(auc = map_df(z$booster,  "auc") %>% 
                         as_tibble() %>% 
                         rename(auc = value) ,
                       optimal_number_of_iter = map(z$booster,  "optimal_number_of_iter")   %>%
                         unlist %>% as_tibble() %>% 
                         rename(optimal_number_of_iter = value))
# a quoi est-ce que ça ressemble
z %>% bind_cols(auc_iter) %>% arrange(-auc)
optimal_number_of_iter <- data.frame( optimal_number_of_iter =  map(z$booster,  "optimal_number_of_iter")  %>% unlist())
# sauvegarder tout 
param_results <- df.params %>% bind_cols(map_df(z$booster,  "auc") %>% rename(auc = value) ) %>% bind_cols(optimal_number_of_iter)  %>% arrange(-auc) 
write_rds(df.params, "df.params.rds")
write_rds(param_results, "param_results.rds")
write_rds(z, "z.rds")
write_rds(alltrain_dummy, "alltrain_dummy.rds")



# Visuellement, quels sont les valeurs des hyperparamètres qui semblent les meilleurs, ceteris paribus?
param_results %>% ggplot(aes(x= max_depth, y = auc)) + geom_smooth() # 6
param_results %>% ggplot(aes(x= colsample_bytree, y = auc)) + geom_smooth() #0.45 
param_results %>% ggplot(aes(x= subsample, y = auc)) + geom_smooth() # 0.3
param_results %>% ggplot(aes(x= min_child_weight, y = auc)) + geom_smooth() # 3
param_results %>% ggplot(aes(x= eta, y = auc)) + geom_smooth() # 0 ou 0.062
param_results %>% ggplot(aes(x= gamma, y = auc)) + geom_smooth() # 0.027


