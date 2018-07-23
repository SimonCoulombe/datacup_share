
# ce script génère  pred_glmnet_train.csv et pred_glmnet_test.csv , qui auraient pu être utilisées dans un stack, mais ça n'a pas servi finalement
nbfold <- 1 # normalement 9 


# préparation des dummies pour caret (default doit être facteur oui/non , pas 0-1 numérique)

label_var <- "Default"

mydb_dummy_train <- mydb_dummy  %>% mutate_all(funs(as.numeric))%>% filter(!is.na(Default)) %>%
  mutate(Default = as.factor(ifelse(Default==0, "non", "oui"))) %>%
  mutate_if(is.integer, as.numeric)

mydb_dummy_train_x <-mydb_dummy_train  %>% mutate_all(funs(as.numeric))%>% select(-Default)
mydb_dummy_train_y <-mydb_dummy_train  %>% mutate_all(funs(as.numeric))%>% select(Default)


mydb_dummy_test <- mydb_dummy %>% mutate_all(funs(as.numeric))  %>% filter(is.na(Default)) %>%
  mutate(Default = as.factor(ifelse(Default==0, "non", "oui")))


#lsite des variables non utilisables (infinies ou NA)

mydb_dummy %>% select_if(function(x) any(!is.finite(x))) %>% 
  summarise_all(funs(sum(!is.finite(.)))) -> finite_mydb
finite_mydb %>% glimpse

#mydb_dummy %>% select(ratio_paiement_on_owed) %>% skimr::skim()
#ratio_paiement_on_owed  pas fini?
mydb_dummy %>% select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.)))) -> NA_mydb

NA_mydb %>% glimpse


# glmnet ----


cctrl1 <- trainControl(
  method="cv", 
  number=nbfold,
  returnResamp="all",
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  allowParallel = TRUE,
  savePredictions = TRUE)
set.seed(1234)

my_glmnet <- train(Default ~. , 
                   data = mydb_dummy_train %>% select( -one_of(colnames(finite_mydb)), - one_of(colnames(NA_mydb))), 
                   method = "glmnet",
                   trControl = cctrl1,
                   metric = "ROC",
                   tuneGrid = expand.grid(alpha = 1,lambda = 0.001)) # seq(0.001,0.1,by = 0.001)


my_glmnet 

# best parameter
my_glmnet$bestTune

# best coefficient
coef(my_glmnet$finalModel, my_glmnet$bestTune$lambda)

pred_caret_glmnet_train <- my_glmnet$pred %>% as_tibble %>% arrange( rowIndex) %>% select(oui)
pred_caret_glmnet_test <- predict(my_glmnet, newdata= mydb_dummy_test, type="prob")

write_csv(pred_caret_glmnet_train, "pred_glmnet_train.csv")
write_csv(pred_caret_glmnet_test, "pred_glmnet_test.csv")


roc_obj <- roc(alltrain_labels, pred_caret_glmnet_train$oui)
auc <- auc(roc_obj) %>% as.tibble() #0.892
