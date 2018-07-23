# ce script génère pred_nn_train.csv et pred_nn_test.csv.  Ces prédictions peuvent ensuite être utilisées dans un ensemble ou dans un stack.

library(keras)
library(caret)
library(tidyverse)
library(DataExplorer)
library(skimr)
library(FactoMineR)
library(factoextra)

# importe et valide ----
important_vars <- read_csv("var_importance.csv") %>% head(20) %>% pull(Feature)

mydb_dummy <- read_rds(path="mydb_dummy.rds") 
mydb_dummy %>% select_if(function(x) any(!is.finite(x))) %>% 
  summarise_all(funs(sum(!is.finite(.)))) -> finite_mydb
finite_mydb %>% glimpse

#mydb_dummy %>% select(ratio_paiement_on_owed) %>% skimr::skim()
#ratio_paiement_on_owed  pas fini?
mydb_dummy %>% select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.)))) -> NA_mydb

NA_mydb %>% glimpse


#nearzero <- nearZeroVar(mydb_dummy)  ## essai pour enlever les shits de eigen crossprod..
#mydb_dummy <- mydb_dummy[,-nearzero]
#important_vars <- read_rds(path="var_importance.rds") %>% head(30) %>% pull(Feature) 
label_var <- "Default"


mydb_dummy_nn <- mydb_dummy %>% select(-Default, -one_of(colnames(finite_mydb)), - one_of(colnames(NA_mydb)))

# PCA ----
combien_de_pca <- 20
myPCA <-PCA( mydb_dummy_nn, scale.unit = TRUE, ncp = combien_de_pca, graph = FALSE) 
predictedPCA <- predict.PCA(myPCA, newdata = mydb_dummy_nn ) 
mydata_pca <-predictedPCA$coord %>% as_tibble
 
# données scalées ----
scale_this <- function(x){
  (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
}

# le modèle donne la même probabilité à tout le monde quand je scale les dummies..
# si je laisse les dummies à 0-1, ça marche.. 

need_scaling <- mydb_dummy_nn %>% 
  summarise_all( function(x) {max(x) -min(x)}) %>%
  gather(key=key, value=value) %>%
  filter(value>1) %>% pull(key)

# On scale puis on fusionne les 30 variables les plus importantes aux 20 PCA.
all_scaled_x <- mydb_dummy_nn %>% 
  mutate_at(vars(need_scaling), funs(scale_this))  %>%
  select(important_vars) %>%
  bind_cols(mydata_pca)
all_labels <- mydb_dummy %>% pull(Default)



# create train, valide, test ----
alltrain_scaled_x <- all_scaled_x[!is.na(all_labels),]
alltrain_labels <- all_labels[!is.na(all_labels)]


trainIndex2 <- caret::createDataPartition(
  alltrain_labels, 
  p = .90, list = FALSE, times = 1)

train_data <-alltrain_scaled_x[trainIndex2, ]
wlist_data <-alltrain_scaled_x[-trainIndex2, ]
test_data <- all_scaled_x[is.na(all_labels),]

train_labels <- alltrain_labels[trainIndex2]
wlist_labels <- alltrain_labels[-trainIndex2]


matrix_alltrain_data <- alltrain_scaled_x %>%  data.matrix()
matrix_train_data <- train_data %>%  data.matrix()
matrix_wlist_data <- wlist_data %>%  data.matrix()
matrix_test_data <- test_data %>%  
  data.matrix()

categ_alltrain_labels <- alltrain_labels %>%  to_categorical()
categ_train_labels <- train_labels %>%  to_categorical()
categ_wlist_labels <- wlist_labels %>% to_categorical()



# Premier réseau - train wlist ----
network <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu", input_shape =dim(train_data)[2]) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 2, activation = "sigmoid")

network %>%
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

network %>% fit(matrix_train_data, 
                categ_train_labels,
                epoch = 8,
                batch_size=16,
                validation_data = list(matrix_wlist_data, categ_wlist_labels))


preds <- network %>% predict(matrix_wlist_data)
plot(preds[,2])                        

# Pred Train - out of fold  ----


get_nn_pred_model <- function(
  train_matrix,   # categorical variables have been replaced by dummies.
  label_matrix,
  nb_fold = 9,
  epoch = 8,
  batch_size = 128
){
  result_get_nn_pred_model <- list()
  
  
  myfolds <- caret::createFolds(
    label_matrix[,2], 
    k = nb_fold, 
    list = FALSE)
  
  myfolds_tibble <- data.frame(fold = myfolds) %>%  as.tibble() %>% mutate(rownum = row_number())
  
  map_result <- seq_len(nb_fold) %>% purrr::map(~{
    function_return <- list()
    message(paste0("calcul du fold ", .x))
    
    matrix_train_data <- train_matrix[myfolds != .x,]
    matrix_wlist_data <- train_matrix[myfolds == .x,]
    
    categ_train_labels <- label_matrix[myfolds != .x,]
    categ_wlist_labels <- label_matrix[myfolds == .x,]
    
    network <- keras_model_sequential() %>%
      layer_dense(units = 20, activation = "relu", input_shape =dim(train_matrix)[2]  ) %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 10, activation = "relu") %>%
      layer_dense(units = 2, activation = "sigmoid")
    
    network %>%
      compile(
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy")
      )
    
    network %>% fit(matrix_train_data, 
                    categ_train_labels,
                    epoch = epoch,
                    batch_size=batch_size,
                    validation_data = list(matrix_wlist_data, categ_wlist_labels))
    
    
    preds <- network %>% predict(matrix_wlist_data)
    
    df.preds <- bind_cols(myfolds_tibble %>% filter(fold ==.x), data.frame(preds = preds[,2]))
    
    return(df.preds)
  })
  
  
  result_get_nn_pred_model <- bind_rows(map_result) %>% arrange(rownum)
  return(result_get_nn_pred_model)
} 

nn_preds_train <-get_nn_pred_model(train_matrix = matrix_alltrain_data,
                                   label_matrix = categ_alltrain_labels,
                                   nb_fold =9,
                                   epoch=8,
                                   batch_size =16)

roc_obj <- roc(alltrain_labels, nn_preds_train$preds)
auc <- auc(roc_obj) %>% as.tibble() #0.900

write_csv(nn_preds_train, "pred_nn_train.csv")


# Pred Test - full sample  ----
network %>% fit(matrix_alltrain_data, 
                categ_alltrain_labels,
                epoch = 8,
                batch_size=16)
                
test_pred <-network %>% predict(matrix_test_data)
write_csv(data.frame(pred_nn= test_pred[,2]), "pred_nn_test.csv")
## fit sur toute les données



preds[,2]

plot(matrix_wlist_data[,1], preds[,2])  # bon un premier modèle qui travaille.. 





