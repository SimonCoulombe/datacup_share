## ce script sauvegarde mylightgbm800.rds, qui peut ensuite être utilisé pour générer des prédictions dans un ensemble ou un stack.
# pas utilisé dans la solution finale


#https://github.com/Microsoft/LightGBM/blob/master/docs/Installation-Guide.rst
#LightGBM uses CMake to build. Run the following commands:
# 
# git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM
# mkdir build ; cd build
# cmake ..
# make -j4
# Note: glibc >= 2.14 is required.
# 
# Also you may want to read gcc Tips.

# library(devtools)
# library(data.table)
# options(devtools.install.args = "--no-multiarch") # if you have 64-bit R only, you can skip this
# install_github("Microsoft/LightGBM", subdir = "R-package")
# install_github('catboost/catboost',subdir='catboost/R-package')

#https://github.com/Microsoft/LightGBM/tree/master/R-package
#https://github.com/Microsoft/LightGBM/blob/master/R-package/demo/cross_validation.R
# hello world mes données -

#install_github("Microsoft/LightGBM", subdir = "R-package")
library(lightgbm)
mydb <- read_rds(path="mydb.rds")  %>% select(-ID_CPTE, -mean_Default_lastCreditRatio_ntile) 
categ_variables <- map_df(mydb %>% select(-Default), class) %>% gather() %>% mutate(rownum = row_number()) %>% mutate(factor = as.numeric(value =="factor")) %>% filter(factor ==1) %>% pull(rownum)

prepared_train_x <- mydb  %>% filter(!is.na(Default)) %>% select(-Default) %>%     lgb.prepare()   
prepared_train_y <- mydb  %>% filter(!is.na(Default)) %>% select(Default)  %>%     lgb.prepare()  %>% pull(Default)
prepared_test_x <- mydb  %>% filter(is.na(Default)) %>% select(-Default)   %>% lgb.prepare() 

matrix_train_x <-prepared_train_x%>% as.matrix(with = FALSE)
matrix_test_x <-prepared_test_x%>% as.matrix(with = FALSE)


lgb_data <- lgb.Dataset(data = matrix_train_x,
                        label = prepared_train_y,
                        categorical_feature = categ_variables)


# We can now train a model

nrounds <- 50
paramss <- list(objective = "binary",
                metric = "AUC",
                min_data_in_leaf = 1,
                learning_rate = 0.01,
                min_hessian = 1,
                #     max_depth =5,
                num_leaves=31,
                feature_fraction= 0.4,
                bagging_fraction = 0.4
                
                )


model <- lgb.train(paramss,
                   lgb_data,
                   nrounds,
                   valids = list(train = lgb_data))
pred <- predict(model, matrix_test_x) # 

# Try to find split_feature: 2
# If you find it, it means it used a categorical feature in the first tree
lgb.dump(model, num_iteration = 1)

write_rds(model, "mylightgbm800.rds")
#model <- read_rds("mylightgbm800.rds")
pred <- predict(model, matrix_test_x) # 
# test CV sur mes données -----

# it means the learning of tree in current iteration should be stop, due to cannot split any more.
# I think this is caused by "min_data_in_leaf":1000, you can set it to a smaller value.

# kfold CV sur mes data ----

print("Running cross validation")
# Do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
lgb.cv(params = paramss,
       data = lgb_data,
       nrounds = nrounds,
       nfold = 5)

# // categorical features tuto-----
#https://github.com/Microsoft/LightGBM/blob/master/R-package/demo/categorical_features_prepare.R

# // cross validation tuto -----
# https://github.com/Microsoft/LightGBM/blob/master/R-package/demo/cross_validation.R

# // catboost - not used ----
#https://tech.yandex.com/catboost/doc/dg/concepts/r-usages-examples-docpage/
# https://github.com/catboost/catboost/blob/master/catboost/tutorials/catboost_r_tutorial.ipynb
