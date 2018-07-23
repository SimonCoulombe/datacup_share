
# libraries ----
library(DataExplorer)
library(xgboost)
library(tidyverse)
library(lubridate)
library(DataExplorer)
library(timeDate)
library(pROC)
library(mgcv)
library(parallel)
library(caret)
library(keras)
set.seed(1234)


# import and bind rows ----
facturation_train <- read_csv(here::here("data", "facturation_train.csv")) 
paiements_train <- read_csv(here::here("data", "paiements_train.csv")) 
transactions_train <- read_csv(here::here("data", "transactions_train.csv"))  
additional_transactions_train <- read_csv(here::here("data", "additional_transactions_train.csv"))  
performance_train <- read_csv(here::here("data", "performance_train.csv")) 
facturation_test <- read_csv(here::here("data", "facturation_test.csv")) 
paiements_test <- read_csv(here::here("data", "paiements_test.csv")) 
transactions_test <- read_csv(here::here("data", "transactions_test.csv"))
additional_transactions_test <- read_csv(here::here("data", "additional_transactions_test.csv"))  
performance_test <- read_csv(here::here("data", "performance_test.csv"))

facturation <- bind_rows(facturation_train, facturation_test)  %>% 
  mutate_if(is.character, as.factor)
paiements <- bind_rows(paiements_train, paiements_test) %>% 
  mutate_if(is.character, as.factor) %>%
  filter(!is.na(TRANSACTION_AMT)) # j'ai 116 zouaves sans paiements

transactions <- bind_rows(transactions_train, transactions_test, 
                          additional_transactions_train, additional_transactions_test) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate(MERCHANT_CITY_NAME  = as.factor(MERCHANT_CITY_NAME ))
performance <- bind_rows(performance_train, performance_test)

sample_solution <- read_csv(here::here("data", "sample_solution.csv"))

# calcul de fold pour les variables target-encodingées..

my50folds <- caret::createFolds(
  performance_train %>% pull(Default), 
  k = 50, list = FALSE)


# Wrangling -----
# //facturation_plus :  un peu de préparation qui va être réutilisée pour créer des variables----

facturation_plus <- facturation %>% 
  mutate(CurrentTotalBalance = pmax(CurrentTotalBalance, CashBalance)) %>%  ## parce que ID_CPTE 62654263 a un mois avec 0 total balance et 1.05 cash balance 
  arrange(ID_CPTE,StatementDate) %>%
  group_by(ID_CPTE) %>%
  mutate(
    rank_from_last_statement = n()-row_number() + 1,
    Q = case_when(
      rank_from_last_statement %in% c(1,2,3,4) ~ 4,
      rank_from_last_statement %in% c(5,6,7) ~ 3,
      rank_from_last_statement %in% c(8,9,10) ~ 2,
      rank_from_last_statement %in% c(11,12,13,14) ~ 1,
      TRUE ~ 5
    ),
    H = case_when(
      rank_from_last_statement %in% c(1,2,3,4,5,6,7) ~ 1,
      rank_from_last_statement %in% c(8,9,10,11,12,13,14) ~ 2,
      TRUE ~ 5),
    lastStatementDate = max(StatementDate),
    is_last_statement = as.numeric(StatementDate ==lastStatementDate),
    nextStatementDate = lead(StatementDate)) %>%
  ungroup() %>%
  mutate(nextStatementDate = if_else(is.na(nextStatementDate) ,
                                     as.Date(StatementDate+30),
                                     nextStatementDate)) %>% # pour le dernier je vais créer une fausse date destatement dans 30 jours pour quantifier les éventuels paiements.
  mutate(CreditRatio = CurrentTotalBalance / CreditLimit,
         CashRatio = CashBalance / CreditLimit,
         CashOnCreditRatio =  ifelse(CashBalance>0,CashBalance / CurrentTotalBalance,0 ))  # tant qu'à y être on va mettre ça tout de suite.

quarter_limits <-
  facturation_plus  %>%  group_by(ID_CPTE, Q) %>% summarise(minStatementDate = min(StatementDate),
                                                  maxStatementDate = max(StatementDate)) %>%
  ungroup()%>%
  mutate(
    minStatementDate = if_else(Q==1, as.Date("2000-01-01"), minStatementDate),
    maxStatementDate = if_else(Q==4, as.Date("2100-01-01"), lead(minStatementDate)-1 )
  ) 
  

# facturation: spread   ----

# une colonne creditratio à chaque rlevé
creditratio_spread <- facturation_plus %>% 
  select(ID_CPTE, CreditRatio, rank_from_last_statement) %>%
  group_by(ID_CPTE) %>%
  rename(CreditRatio_t_minus_ = rank_from_last_statement) %>% 
  spread(key= CreditRatio_t_minus_, value = CreditRatio, sep="") %>%
  ungroup()
  
# la différence de balance totale à chaque relevé
delta_monthly_CurrentTotalBalance_spread <- facturation_plus %>% 
  select(ID_CPTE, CurrentTotalBalance, rank_from_last_statement) %>%
  group_by(ID_CPTE) %>%
  mutate(delta_monthly_CurrentTotalBalance =  CurrentTotalBalance- lag(CurrentTotalBalance)) %>%
  select(-CurrentTotalBalance)%>%
  rename(delta_monthly_CurrentTotalBalance_t_minus_ = rank_from_last_statement) %>% 
  spread(key= delta_monthly_CurrentTotalBalance_t_minus_, value = delta_monthly_CurrentTotalBalance, sep="") %>%
  ungroup()

# la balance total en dollar à chaque relevé
CurrentTotalBalance_spread <- facturation_plus %>% 
  select(ID_CPTE, CurrentTotalBalance, rank_from_last_statement) %>%
  group_by(ID_CPTE) %>%
  rename(CurrentTotalBalance_t_minus_ = rank_from_last_statement) %>% 
  spread(key= CurrentTotalBalance_t_minus_, value = CurrentTotalBalance, sep="") %>%
  ungroup()


  

facturation_spread <- creditratio_spread %>% 
  left_join(delta_monthly_CurrentTotalBalance_spread) %>%
  left_join(CurrentTotalBalance_spread)


# facturation - pct_paid_until_next_statement   ----
pct_paid_until_next_statement <-
  facturation_plus  %>% 
  left_join(facturation_plus %>% 
              left_join(paiements %>%  # on a tout les bills 
                          mutate(TRANSACTION_DT = as.Date(TRANSACTION_DTTM)) %>%
                          select(ID_CPTE,TRANSACTION_DT, TRANSACTION_AMT )) %>%
              filter(StatementDate <= TRANSACTION_DT, nextStatementDate >TRANSACTION_DT) %>%  # paiements dd'ici le prochain relevé
              group_by(ID_CPTE,StatementDate,CurrentTotalBalance) %>%
              summarise(total_paid_until_next_statement = sum(TRANSACTION_AMT)) %>%
              ungroup()) %>%
  mutate(total_paid_until_next_statement = replace_na(total_paid_until_next_statement,0)) %>%  # si c'est NA c'est parce qu'il y a pas eu de paiement
  mutate(pct_paid_until_next_statement = ifelse( CurrentTotalBalance ==0 |  total_paid_until_next_statement>CurrentTotalBalance , 1, total_paid_until_next_statement/CurrentTotalBalance)) %>%
  select(ID_CPTE, StatementDate, is_last_statement, total_paid_until_next_statement, pct_paid_until_next_statement, rank_from_last_statement)

# quel pourcentage de ta balance a été payé dans les 30 prochains jours? (1 colonne par relevé)
pct_paid_until_next_statement_spread <- pct_paid_until_next_statement %>% 
  select(ID_CPTE, pct_paid_until_next_statement, rank_from_last_statement) %>%
  group_by(ID_CPTE) %>%
  rename(pct_paid_until_next_statement_t_minus_ = rank_from_last_statement) %>% 
  spread(key= pct_paid_until_next_statement_t_minus_, value = pct_paid_until_next_statement, sep="") %>%
  ungroup()
  


# quelle est la moyenne, max, min , skew et kurtosis du pourcentage calculé ci haut?
pct_paid_until_next_statement_summary <- 
  pct_paid_until_next_statement %>% 
  filter(is_last_statement ==0) %>%  # le dernier statemeent est jamais payé car pas le temps..
  group_by(ID_CPTE)%>%
  summarise(
    mean_pct_paid_until_next_statement = mean(pct_paid_until_next_statement, na.rm = TRUE) ,
    max_pct_paid_until_next_statement = max(pct_paid_until_next_statement, na.rm = TRUE),
    min_pct_paid_until_next_statement = min(pct_paid_until_next_statement, na.rm = TRUE),
    kurtosis_pct_paid_until_next_statement = kurtosis(pct_paid_until_next_statement, na.rm= TRUE),
    skewness_pct_paid_until_next_statement = skewness(pct_paid_until_next_statement, na.rm= TRUE)) %>%
    #last_pct_paid_until_next_statement = max(pct_paid_until_next_statement * (StatementDate == max(StatementDate)), na.rm = TRUE)) %>% # remplaceé par t_minus_2
  ungroup() %>%
  left_join(pct_paid_until_next_statement_spread)
  

# quelle est la tendance dans le pourcentage que tu a payé? (1 régression par personne)

pct_paid_until_next_statement_nest <- pct_paid_until_next_statement %>%
  filter(is_last_statement ==0) %>%  # le dernier statemeent est jamais payé car pas le temps..
  group_by(ID_CPTE) %>%
  mutate(rownum = row_number())%>%
  nest()

trend_pct_paid_until_next_statement_temp <- pct_paid_until_next_statement_nest %>%
  mutate(estimate = map(data, function(X){
    lm( pct_paid_until_next_statement ~ rownum, data=X) %>%
      broom::tidy() %>% as_tibble %>% filter(term== "rownum") %>% select(estimate)
  } ) )

trend_pct_paid_until_next_statement <- trend_pct_paid_until_next_statement_temp %>% mutate(trend_pct_paid_until_next_statement = map(estimate, function(X){ unlist(X) }) %>% as.numeric) %>% select(ID_CPTE, trend_pct_paid_until_next_statement)

write_rds(trend_pct_paid_until_next_statement, "trend_pct_paid_until_next_statement.rds")
 trend_pct_paid_until_next_statement %>% plot_histogram()

#trend_pct_paid_until_next_statement <- read_rds("trend_pct_paid_until_next_statement.rds")
 
  
# pour lesfactures avec un solde positives, ça prenait combien de temps avant un premier paiement? avec un paiement complet?
bills <- facturation_plus %>% 
  filter(CurrentTotalBalance >0) %>%
  arrange(ID_CPTE, StatementDate)

paiements_post_statement <-bills %>%
  select(ID_CPTE, StatementDate) %>%
  left_join(paiements %>%  # on a tout les bills 
                    mutate(TRANSACTION_DT = as.Date(TRANSACTION_DTTM)) %>%
                    select(ID_CPTE,TRANSACTION_DT, TRANSACTION_AMT )) %>%
  filter( StatementDate <= TRANSACTION_DT ) # on perd les bills qui on pas de transaction date, ce qui exploque que je parte d'un bill %>% left_join
  
paiements_post_statement %>% distinct(ID_CPTE, StatementDate)

# https://stackoverflow.com/questions/6668963/how-to-prevent-ifelse-from-turning-date-objects-into-numeric-objects
# You may use dplyr::if_else.
# 
# From dplyr 0.5.0 release notes: "[if_else] have stricter semantics that ifelse(): the true and  false arguments must be the same type. This gives a less surprising return type, and preserves S3 vectors like dates" .
next_paiements <- 
  paiements_post_statement %>%
  group_by(ID_CPTE, StatementDate) %>%
  arrange(TRANSACTION_DT) %>%
  slice(1) %>% 
  ungroup() %>%
  select(ID_CPTE, StatementDate, next_paiement_date = TRANSACTION_DT)

#delai_premier_paiement <- next_paiements  %>%   mutate(delai_premier_paiement =next_paiement_date -StatementDate  )
# TODO : si pas payé, trouver depuis combien de temps pas payé.?
# TODO: réviser ma méthodologie, c'est normal que ton délai baisse avec le temps pour les non-NA, car t'as moins de temps avantla fin..

next_full_paiements <- 
  bills %>% 
  select(ID_CPTE, StatementDate, CurrentTotalBalance) %>%  # liste des bills à laquelle on merge les paiements si il y en a
  left_join(paiements_post_statement) %>%
  group_by(ID_CPTE, StatementDate) %>%
  arrange(TRANSACTION_DT) %>% 
  mutate(cumsum = cumsum(TRANSACTION_AMT)) %>%
  filter(cumsum >= CurrentTotalBalance) %>%
  slice(1) %>%
  ungroup() %>%
  select(ID_CPTE, StatementDate, full_paiement_date = TRANSACTION_DT)

has_full_paiement <- next_full_paiements %>% distinct(ID_CPTE) %>% mutate(has_full_paiement=1)

# bills_outcome a encore une ligne par facture
bills_outcome <- 
  bills %>% 
  left_join(next_paiements) %>%
  left_join(next_full_paiements) %>%
  mutate(delai_next_paiement =next_paiement_date    - StatementDate,
         delai_full_paiement =full_paiement_date - StatementDate )  %>%
  mutate_if(is.difftime, as.numeric) %>%
  mutate(time_to_last_statement_plus_60 = lastStatementDate-StatementDate+ 60) %>% 
  group_by(ID_CPTE) %>%  # TODO  si pas de future paiement, on remplace par le maximum entre A) le maximum de cette personne B) le temps d'ici last statementdate+60 # je pogne 460 jours..
  mutate(max_delai_next_paiement = max(delai_next_paiement, na.rm=TRUE)) %>%
  mutate(max_delai_full_paiement = max(delai_full_paiement, na.rm=TRUE)) %>%
  ungroup()  %>%
  mutate(
         
    delai_next_paiement_capped = ifelse(  !is.na(delai_next_paiement), 
                                    delai_next_paiement ,
                                    pmax(max_delai_next_paiement, time_to_last_statement_plus_60   )),
    delai_full_paiement_capped = ifelse(  !is.na(delai_full_paiement), 
                                   delai_full_paiement ,
                                   pmax(max_delai_full_paiement, time_to_last_statement_plus_60   )))

  
#bills_outcome_summary fait un sommaire des variables :  pct des factures qui ne sont pas suivis de paiement,
#pct  des factures qui n'ont jamais été payées en entier, 
# le délai maximal et moyen avant le premier paiement/ paiement complet ,etc..
# etc..



bills_outcome_summary <-
  bills_outcome %>% 
  filter(is_last_statement ==0 ) %>% #TODO  filter enleverl e dernier bill
  group_by(ID_CPTE) %>%
  summarise(
    count_bills = n(),
    pct_no_next_paiement = sum( is.na(delai_next_paiement)) / n(),
    pct_no_full_paiement = sum( is.na(delai_full_paiement)) / n(),
    mean_delai_next_paiement = mean(delai_next_paiement, na.rm = TRUE) ,
    mean_delai_full_paiement = mean(delai_full_paiement, na.rm = TRUE),
    max_delai_next_paiement = max(delai_next_paiement, na.rm = TRUE)%>% ifelse(is.infinite(.),NA_real_, .),
    max_delai_full_paiement = max(delai_full_paiement, na.rm = TRUE)%>% ifelse(is.infinite(.),NA_real_, .),
    kurtosis_delai_next_paiement = kurtosis(delai_next_paiement, na.rm= TRUE),
    kurtosis_delai_full_paiement = kurtosis(delai_full_paiement, na.rm= TRUE),
    skewness_delai_next_paiement = skewness(delai_next_paiement, na.rm= TRUE),
    skewness_delai_full_paiement = skewness(delai_full_paiement, na.rm= TRUE),
    last_delai_next_paiement = max(delai_next_paiement * (PERIODID_MY == max(PERIODID_MY)), na.rm = TRUE)%>% ifelse(is.infinite(.),NA_real_, .),
    last_delai_full_paiement = max(delai_full_paiement* (PERIODID_MY == max(PERIODID_MY)), na.rm = TRUE)%>% ifelse(is.infinite(.),NA_real_, .)
    )


# calcul de la tendance dans le délai de paiement (1 régression par personne)
bills_outcome_full_nest <- bills_outcome %>%
  filter(is_last_statement ==0 ) %>%
  filter(!is.na(delai_full_paiement_capped)) %>%
  group_by(ID_CPTE) %>%
  nest()

bills_outcome_next_nest <- bills_outcome %>%
  filter(is_last_statement ==0 ) %>%
  filter(!is.na(delai_next_paiement_capped)) %>%
  group_by(ID_CPTE) %>%
  nest()
# calcul de la tendance dans le délai de paiement (1 régression par personne)

trend_delai_full_paiement_temp <- bills_outcome_full_nest %>%
  mutate(estimate = map(data, function(X){
    lm( delai_full_paiement_capped ~ StatementDate, data=X) %>%
      broom::tidy() %>% as_tibble %>% filter(term== "StatementDate") %>% select(estimate)
  } ) )

trend_delai_next_paiement_temp <- bills_outcome_next_nest %>%
  mutate(estimate = map(data, function(X){
    lm( delai_next_paiement_capped ~ StatementDate, data=X) %>%
      broom::tidy() %>% as_tibble %>% filter(term== "StatementDate") %>% select(estimate)

  } ) )


trend_delai_full_paiement <- trend_delai_full_paiement_temp %>% mutate(trend_delai_full_paiement = map(estimate, function(X){ unlist(X) }) %>% as.numeric) %>% select(ID_CPTE, trend_delai_full_paiement)

trend_delai_next_paiement <- trend_delai_next_paiement_temp %>% mutate(trend_delai_next_paiement = map(estimate, function(X){ unlist(X) }) %>% as.numeric) %>% select(ID_CPTE, trend_delai_next_paiement)

write_rds(trend_delai_full_paiement, "trend_delai_full_paiement.rds")
write_rds(trend_delai_next_paiement, "trend_delai_next_paiement.rds")

# trend_delai_full_paiement <- read_rds("trend_delai_full_paiement.rds")
# trend_delai_next_paiement <- read_rds("trend_delai_next_paiement.rds")


# facturation: trend de credit ratio ----
# ici on va calculer une régression par personne pour voir la tendance dans ton credit ratio  


facturation_nest <- facturation %>%
  group_by(ID_CPTE) %>%
  mutate(CreditRatio = CurrentTotalBalance / CreditLimit,
         CashRatio = CashBalance / CreditLimit,
         CashOnCreditRatio =  ifelse(CashBalance>0,CashBalance / CurrentTotalBalance,0 )) %>%
  nest()


trend_CreditRatio_temp <- facturation_nest %>%
  mutate(
    estimate = map(data, function(X){
      lm( CreditRatio ~ StatementDate, data=X) %>%
        broom::tidy() %>% as_tibble %>% filter(term== "StatementDate") %>% select(estimate)}))

trend_CashRatio_temp <- facturation_nest %>%
  mutate(
    estimate = map(data, function(X){
      lm( CashRatio ~ StatementDate, data=X) %>%
        broom::tidy() %>% as_tibble %>% filter(term== "StatementDate") %>% select(estimate)}))

# TODO régler cette erreur
#Error in mutate_impl(.data, dots) : Evaluation error: NA/NaN/Inf in 'y'
# trend_CashOnCreditRatio_temp <- facturation_nest %>%
#   mutate(
#     estimate = map(data, function(X){
#       lm( CashOnCreditRatio ~ StatementDate, data=X) %>%
#         broom::tidy() %>% as_tibble %>% filter(term== "StatementDate") %>% select(estimate)}))
#


trend_CreditRatio <- trend_CreditRatio_temp %>% mutate(trend_CreditRatio = map(estimate, function(X){ unlist(X) }) %>% as.numeric) %>% select(ID_CPTE, trend_CreditRatio)
trend_CashRatio <- trend_CashRatio_temp %>% mutate(trend_CashRatio = map(estimate, function(X){ unlist(X) }) %>% as.numeric) %>% select(ID_CPTE, trend_CashRatio)

write_rds(trend_CreditRatio, "trend_CreditRatio.rds")
write_rds(trend_CashRatio, "trend_CashRatio.rds")
# trend_CreditRatio <- read_rds("trend_CreditRatio.rds")
# trend_CashRatio <- read_rds("trend_CashRatio.rds")

# facturation (trimestre) summary ----

# On fait des variables résumés de tes factures pour chacun des trimestres
# la star c'est la fonction unite entre le gather et le spread.
facturation_quarter_summary <- facturation_plus %>% group_by(ID_CPTE, Q) %>%
  summarise(
    sum_CurrentTotalBalance = sum(CurrentTotalBalance),
    count_positive_CurrentTotalBalance = sum(CurrentTotalBalance>0),
    count_positive_CashBalance = sum(CashBalance>0),
    count_positive_DelqCycle = sum(DelqCycle>0),
    sumDelqCycle = sum(DelqCycle),
    sumCreditLimit = sum(CreditLimit),
    sumCashBalance = sum(CashBalance),
    sumCurrentTotalBalance = sum(CurrentTotalBalance),
    sumCreditRatio = sum(CreditRatio),
    sumCashRatio = sum(CashRatio),
    sumCashOnCreditRatio = sum(CashOnCreditRatio),
    maxDelqCycle = max(DelqCycle),
    maxCreditLimit = max(CreditLimit),
    maxCashBalance = max(CashBalance),
    maxCurrentTotalBalance = max(CurrentTotalBalance),
    maxCreditRatio = max(CreditRatio),
    maxCashRatio = max(CashRatio),
    maxCashOnCreditRatio = max(CashOnCreditRatio),
    meanDelqCycle = mean(DelqCycle),
    meanCreditLimit = mean(CreditLimit),
    meanCashBalance = mean(CashBalance),
    meanCurrentTotalBalance = mean(CurrentTotalBalance),
    meanCreditRatio = mean(CreditRatio),
    meanCashRatio = mean(CashRatio),
    meanCashOnCreditRatio = mean(CashOnCreditRatio),
    minDelqCycle = min(DelqCycle),
    minCreditLimit = min(CreditLimit),
    minCashBalance = min(CashBalance),
    minCurrentTotalBalance = min(CurrentTotalBalance),
    minCreditRatio = min(CreditRatio),
    minCashRatio = min(CashRatio),
    minCashOnCreditRatio = min(CashOnCreditRatio),
    kurtosisDelqCycle = kurtosis(DelqCycle),
    kurtosisCreditLimit = kurtosis(CreditLimit),
    kurtosisCashBalance = kurtosis(CashBalance),
    kurtosisCurrentTotalBalance = kurtosis(CurrentTotalBalance),
    kurtosisCreditRatio = kurtosis(CreditRatio),
    kurtosisCashRatio = kurtosis(CashRatio),
    kurtosisCashOnCreditRatio = kurtosis(CashOnCreditRatio),
    skewnessDelqCycle = skewness(DelqCycle),
    skewnessCreditLimit = skewness(CreditLimit),
    skewnessCashBalance = skewness(CashBalance),
    skewnessCurrentTotalBalance = skewness(CurrentTotalBalance),
    skewnessCreditRatio = skewness(CreditRatio),
    skewnessCashRatio = skewness(CashRatio),
sdDelqCycle = sd(DelqCycle),
sdCreditLimit = sd(CreditLimit),
sdCashBalance = sd(CashBalance),
sdCurrentTotalBalance = sd(CurrentTotalBalance),
sdCreditRatio = sd(CreditRatio),
sdCashRatio = sd(CashRatio),
p25DelqCycle = quantile(DelqCycle,0.25),
p25CreditLimit = quantile(CreditLimit,0.25),
p25CashBalance = quantile(CashBalance,0.25),
p25CurrentTotalBalance = quantile(CurrentTotalBalance,0.25),
p25CreditRatio = quantile(CreditRatio,0.25),
p25CashRatio = quantile(CashRatio,0.25),
p75DelqCycle = quantile(DelqCycle,0.75),
p75CreditLimit = quantile(CreditLimit,0.75),
p75CashBalance = quantile(CashBalance,0.75),
p75CurrentTotalBalance = quantile(CurrentTotalBalance,0.75),
p75CreditRatio = quantile(CreditRatio,0.75),
p75CashRatio = quantile(CashRatio,0.75))  %>%
  ungroup()  %>%
  gather(-ID_CPTE, -Q, key="key", value = "value") %>%
  unite(Var, key, Q, sep="_Q") %>%
  spread (Var, value)

# facturation (semester) summary ----
# on recommence au niveau du semestre
facturation_semester_summary <- facturation_plus %>% group_by(ID_CPTE, H) %>%
  summarise(
    sum_CurrentTotalBalance = sum(CurrentTotalBalance),
    count_positive_CurrentTotalBalance = sum(CurrentTotalBalance>0),
    count_positive_CashBalance = sum(CashBalance>0),
    count_positive_DelqCycle = sum(DelqCycle>0),
    sumDelqCycle = sum(DelqCycle),
    sumCreditLimit = sum(CreditLimit),
    sumCashBalance = sum(CashBalance),
    sumCurrentTotalBalance = sum(CurrentTotalBalance),
    sumCreditRatio = sum(CreditRatio),
    sumCashRatio = sum(CashRatio),
    sumCashOnCreditRatio = sum(CashOnCreditRatio),
    maxDelqCycle = max(DelqCycle),
    maxCreditLimit = max(CreditLimit),
    maxCashBalance = max(CashBalance),
    maxCurrentTotalBalance = max(CurrentTotalBalance),
    maxCreditRatio = max(CreditRatio),
    maxCashRatio = max(CashRatio),
    maxCashOnCreditRatio = max(CashOnCreditRatio),
    meanDelqCycle = mean(DelqCycle),
    meanCreditLimit = mean(CreditLimit),
    meanCashBalance = mean(CashBalance),
    meanCurrentTotalBalance = mean(CurrentTotalBalance),
    meanCreditRatio = mean(CreditRatio),
    meanCashRatio = mean(CashRatio),
    meanCashOnCreditRatio = mean(CashOnCreditRatio),
    minDelqCycle = min(DelqCycle),
    minCreditLimit = min(CreditLimit),
    minCashBalance = min(CashBalance),
    minCurrentTotalBalance = min(CurrentTotalBalance),
    minCreditRatio = min(CreditRatio),
    minCashRatio = min(CashRatio),
    minCashOnCreditRatio = min(CashOnCreditRatio),
    kurtosisDelqCycle = kurtosis(DelqCycle),
    kurtosisCreditLimit = kurtosis(CreditLimit),
    kurtosisCashBalance = kurtosis(CashBalance),
    kurtosisCurrentTotalBalance = kurtosis(CurrentTotalBalance),
    kurtosisCreditRatio = kurtosis(CreditRatio),
    kurtosisCashRatio = kurtosis(CashRatio),
    kurtosisCashOnCreditRatio = kurtosis(CashOnCreditRatio),
    skewnessDelqCycle = skewness(DelqCycle),
    skewnessCreditLimit = skewness(CreditLimit),
    skewnessCashBalance = skewness(CashBalance),
    skewnessCurrentTotalBalance = skewness(CurrentTotalBalance),
    skewnessCreditRatio = skewness(CreditRatio),
    skewnessCashRatio = skewness(CashRatio),
    sdDelqCycle = sd(DelqCycle),
    sdCreditLimit = sd(CreditLimit),
    sdCashBalance = sd(CashBalance),
    sdCurrentTotalBalance = sd(CurrentTotalBalance),
    sdCreditRatio = sd(CreditRatio),
    sdCashRatio = sd(CashRatio),
    p25DelqCycle = quantile(DelqCycle,0.25),
    p25CreditLimit = quantile(CreditLimit,0.25),
    p25CashBalance = quantile(CashBalance,0.25),
    p25CurrentTotalBalance = quantile(CurrentTotalBalance,0.25),
    p25CreditRatio = quantile(CreditRatio,0.25),
    p25CashRatio = quantile(CashRatio,0.25),
    p75DelqCycle = quantile(DelqCycle,0.75),
    p75CreditLimit = quantile(CreditLimit,0.75),
    p75CashBalance = quantile(CashBalance,0.75),
    p75CurrentTotalBalance = quantile(CurrentTotalBalance,0.75),
    p75CreditRatio = quantile(CreditRatio,0.75),
    p75CashRatio = quantile(CashRatio,0.75)) %>%
  ungroup()  %>%
  gather(-ID_CPTE, -H, key="key", value = "value") %>%
  unite(Var, key, H, sep="_H") %>%
  spread (Var, value)

# Facturation_summary ----
# on recommence pour l'année entière.
# ensuite on greffe aussi les variables de facturations générées plus tôt.
# puis on génère la première variable "target encodée" : mean_Default_lastCreditRatio_ntile



facturation_summary_temp <- facturation_plus %>% group_by(ID_CPTE) %>%
  summarise(
    count_facture = n(),
    sum_CurrentTotalBalance = sum(CurrentTotalBalance),
    count_positive_CurrentTotalBalance = sum(CurrentTotalBalance>0),
    count_positive_CashBalance = sum(CashBalance>0),
    count_positive_DelqCycle = sum(DelqCycle>0),
    sumDelqCycle = sum(DelqCycle),
    sumCreditLimit = sum(CreditLimit),
    sumCashBalance = sum(CashBalance),
    sumCurrentTotalBalance = sum(CurrentTotalBalance),
    sumCreditRatio = sum(CreditRatio),
    sumCashRatio = sum(CashRatio),
    sumCashOnCreditRatio = sum(CashOnCreditRatio),
    maxDelqCycle = max(DelqCycle),
    maxCreditLimit = max(CreditLimit),
    maxCashBalance = max(CashBalance),
    maxCurrentTotalBalance = max(CurrentTotalBalance),
    maxCreditRatio = max(CreditRatio),
    maxCashRatio = max(CashRatio),
    maxCashOnCreditRatio = max(CashOnCreditRatio),
    meanDelqCycle = mean(DelqCycle),
    meanCreditLimit = mean(CreditLimit),
    meanCashBalance = mean(CashBalance),
    meanCurrentTotalBalance = mean(CurrentTotalBalance),
    meanCreditRatio = mean(CreditRatio),
    meanCashRatio = mean(CashRatio),
    meanCashOnCreditRatio = mean(CashOnCreditRatio),
    minDelqCycle = min(DelqCycle),
    minCreditLimit = min(CreditLimit),
    minCashBalance = min(CashBalance),
    minCurrentTotalBalance = min(CurrentTotalBalance),
    minCreditRatio = min(CreditRatio),
    minCashRatio = min(CashRatio),
    minCashOnCreditRatio = min(CashOnCreditRatio),
    kurtosisDelqCycle = kurtosis(DelqCycle),
    kurtosisCreditLimit = kurtosis(CreditLimit),
    kurtosisCashBalance = kurtosis(CashBalance),
    kurtosisCurrentTotalBalance = kurtosis(CurrentTotalBalance),
    kurtosisCreditRatio = kurtosis(CreditRatio),
    kurtosisCashRatio = kurtosis(CashRatio),
    kurtosisCashOnCreditRatio = kurtosis(CashOnCreditRatio),
    skewnessDelqCycle = skewness(DelqCycle),
    skewnessCreditLimit = skewness(CreditLimit),
    skewnessCashBalance = skewness(CashBalance),
    skewnessCurrentTotalBalance = skewness(CurrentTotalBalance),
    skewnessCreditRatio = skewness(CreditRatio),
    skewnessCashRatio = skewness(CashRatio),
    skewnessCreditRatio = skewness(CashOnCreditRatio),
    sdDelqCycle = sd(DelqCycle),
    sdCreditLimit = sd(CreditLimit),
    sdCashBalance = sd(CashBalance),
    sdCurrentTotalBalance = sd(CurrentTotalBalance),
    sdCreditRatio = sd(CreditRatio),
    sdCashRatio = sd(CashRatio),
    p25DelqCycle = quantile(DelqCycle,0.25),
    p25CreditLimit = quantile(CreditLimit,0.25),
    p25CashBalance = quantile(CashBalance,0.25),
    p25CurrentTotalBalance = quantile(CurrentTotalBalance,0.25),
    p25CreditRatio = quantile(CreditRatio,0.25),
    p25CashRatio = quantile(CashRatio,0.25),
    p75DelqCycle = quantile(DelqCycle,0.75),
    p75CreditLimit = quantile(CreditLimit,0.75),
    p75CashBalance = quantile(CashBalance,0.75),
    p75CurrentTotalBalance = quantile(CurrentTotalBalance,0.75),
    p75CreditRatio = quantile(CreditRatio,0.75),
    p75CashRatio = quantile(CashRatio,0.75),
    lastDelqCycle = max(DelqCycle *( PERIODID_MY == max(PERIODID_MY))),
    lastCreditLimit = max(CreditLimit *( PERIODID_MY == max(PERIODID_MY))),
    lastCashBalance = max(CashBalance *( PERIODID_MY == max(PERIODID_MY))),
    lastCurrentTotalBalance = max(CurrentTotalBalance *( PERIODID_MY == max(PERIODID_MY))),
    lastCreditRatio = max(CreditRatio *( PERIODID_MY == max(PERIODID_MY))),
    lastCashRatio = max(CashRatio *( PERIODID_MY == max(PERIODID_MY))),
    lastCashOnCreditRatio = max(CashOnCreditRatio *( PERIODID_MY == max(PERIODID_MY))),
    firstDelqCycle = max(DelqCycle *( PERIODID_MY == min(PERIODID_MY))),
    firstCreditLimit = max(CreditLimit *( PERIODID_MY == min(PERIODID_MY))),
    firstCashBalance = max(CashBalance *( PERIODID_MY == min(PERIODID_MY))),
    firstCurrentTotalBalance = max(CurrentTotalBalance *( PERIODID_MY == min(PERIODID_MY))),
    firstCreditRatio = max(CreditRatio *( PERIODID_MY == min(PERIODID_MY))),
    firstCashRatio = max(CashRatio *( PERIODID_MY == min(PERIODID_MY))),
    firstCashOnCreditRatio = max(CashOnCreditRatio *( PERIODID_MY == min(PERIODID_MY))),
    atnewmaxCreditLimit = as.integer(lastCreditLimit== maxCreditLimit & lastCreditLimit > minCreditLimit),
    atnewminCreditLimit = as.integer(lastCreditLimit== minCreditLimit & lastCreditLimit < maxCreditLimit),
    atnewmaxCreditRatio = as.integer(lastCreditRatio== maxCreditRatio & lastCreditRatio > minCreditRatio),
    atnewminCreditRatio = as.integer(lastCreditRatio== minCreditRatio & lastCreditRatio < maxCreditRatio),
    atnewmaxCashRatio = as.integer(lastCashRatio== maxCashRatio & lastCashRatio > minCashRatio),
    atnewminCashRatio = as.integer(lastCashRatio== minCashRatio & lastCashRatio < maxCashRatio),
    annualdeltaCreditRatio = lastCreditRatio -firstCreditRatio,
    annualdeltaCashRatio = lastCashRatio -firstCashRatio,
    annualdeltaCurrentTotalBalance = lastCurrentTotalBalance - firstCurrentTotalBalance,
    annualdeltaCreditLimit = lastCreditLimit - firstCreditLimit,
    last_on_max_CreditRatio = ifelse(maxCreditRatio>0, lastCreditRatio/maxCreditRatio, 1 ), # si jamais de solde ça plante
    lastCreditRatio_times_lastDelqCycle = lastCreditRatio *lastDelqCycle) %>%
  ungroup() %>%
  mutate(lastCreditRatio_ntile = as.factor(ntile(lastCreditRatio,10))) %>% 
  left_join(facturation_spread)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!! Hic sunt dracones / Here be Dragons. !!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# J'ai mal fait mon target encoding et il y a du leakage..  il faudrait plutôt faire la moyenne de default pour les gens qui ne sont pas toi..
# sinon quand je fais du out of fold je vais quand meme voir que 100% ou 0% de ton groupe a fait default. 
# je l'ai bien fait pour city_mean_default, voir plus bas.
# bref, en attendant de réparer  mean_Default_lastCreditRatio_ntile il ne faut pas l'utiliser



# facturation target encoding de credit ratio (moyenne de default pour le groupe train)
facturation_summary <- facturation_summary_temp %>%
  left_join(facturation_summary_temp %>% 
  select(ID_CPTE, lastCreditRatio_ntile) %>%
  inner_join(performance_train %>% select(ID_CPTE, Default)) %>%
  group_by(lastCreditRatio_ntile) %>%
  summarise(mean_Default_lastCreditRatio_ntile = mean(Default))) %>%
  left_join(facturation_semester_summary) %>% 
  left_join(facturation_quarter_summary) %>%
  mutate(CreditRatio_ratio_Q4Q1 = ifelse(meanCreditRatio_Q1>1, meanCreditRatio_Q4 / meanCreditRatio_Q1, 1))

# Paiements_summary ----
# TODO utilise les délais entre les relevés au lieu du mois, comme pour les pcT_paid_untilnext_statement ----

##  min /max / mean / last paiement mensuel
paiements_months <- paiements %>%   
  mutate(month = floor_date(as.Date(TRANSACTION_DTTM), "month")) %>%
  group_by(ID_CPTE,month) %>% 
  mutate(monthly_paiement_amt = sum(TRANSACTION_AMT)) %>%
  ungroup() %>%
  group_by(ID_CPTE) %>%
  summarise(
    count_monthly_paiement_amt = n(),
    mean_monthly_paiement_amt = mean(monthly_paiement_amt),
    max_monthly_paiement_amt = max(monthly_paiement_amt),
    min_monthly_paiement_amt = min(monthly_paiement_amt),
    last_monthly_paiement_amt = max( monthly_paiement_amt * (month == max(month)))) %>%
  ungroup()
  
# ok on fusionne ici tout ce qu'on sait sur les paiements
# on crée aussi qq variables résumé sur les TRANSACTION_AMT
paiements_summary <- paiements %>%   
  group_by(ID_CPTE)  %>%
  summarise(count_paiements = n(),
            last_paiement_date = as.Date(max(TRANSACTION_DTTM)),
            min_paiement_amt = min(TRANSACTION_AMT),
            mean_paiement_amt = mean(TRANSACTION_AMT),
            max_paiement_amt = max(TRANSACTION_AMT),
            sum_paiement_amt = sum(TRANSACTION_AMT),
            kurtosis_paiement_amt = kurtosis(TRANSACTION_AMT),
            skewness_paiement_amt = skewness(TRANSACTION_AMT),
            count_paiement_reversal_flag  =sum(PAYMENT_REVERSAL_XFLG == "N")) %>%
  ungroup() %>%
  mutate(has_paiement_data = 1) %>%
  left_join(paiements_months)

  
# Transactions_summary ----

#il y a beaucoup de catégories, pays et villes  de marchands..
# je vais essayer de regrouper les catégories selon le décike fréquence (petites villes, moyenne ville, grosses  villes )

transactions_plus <- transactions %>%  left_join(quarter_limits) %>%
  mutate(transaction_date = as.Date(TRANSACTION_DTTM ))  %>%
  filter(transaction_date >= minStatementDate ,  maxStatementDate >= transaction_date)



MERCHANT_CATEGORY_XCD_ntile <- transactions %>% group_by(MERCHANT_CATEGORY_XCD ) %>% 
  count() %>% ungroup() %>% mutate(MERCHANT_CATEGORY_XCD_ntile = as.factor(ntile(n,10))) %>% select(-n)

MERCHANT_CITY_NAME_ntile <- transactions %>% group_by(MERCHANT_CITY_NAME ) %>% 
  count() %>% ungroup() %>% mutate(MERCHANT_CITY_NAME_ntile = as.factor(ntile(n,10))) %>% select(-n)
MERCHANT_COUNTRY_XCD_ntile <-transactions %>% group_by(MERCHANT_COUNTRY_XCD ) %>% 
  count() %>% ungroup() %>% mutate(MERCHANT_COUNTRY_XCD_ntile = as.factor(ntile(n,10))) %>% select(-n)
# pct de DECISION_XCD par personne
transactions_temp<- transactions %>% 
  left_join(MERCHANT_CATEGORY_XCD_ntile) %>% 
  left_join(MERCHANT_CITY_NAME_ntile) %>%
  left_join(MERCHANT_COUNTRY_XCD_ntile) %>% 
  ungroup()

# Ok, la je veux voir quelle proportion de ton argent a été dépensé dans chaque catégorie 

get_sum_pct <- function( data, var_name){
  quos_var_name <- rlang::sym(var_name)
  data %>% group_by(ID_CPTE, !!quos_var_name) %>%
    summarise(mysum = sum(TRANSACTION_AMT)) %>% 
    ungroup() %>%
    group_by(ID_CPTE ) %>%
    mutate(pct  = mysum / sum(mysum))  %>%
    ungroup()%>%
    select(ID_CPTE, !!quos_var_name, pct) %>%
    spread(key=!!quos_var_name, value = pct, fill = 0, sep = paste0("pct_sum_")) # , quos_var_name
}

DECISION_XCD <- get_sum_pct(transactions_temp, "DECISION_XCD")
DECISION_XCD <- get_sum_pct(transactions_temp, "DECISION_XCD")
MERCHANT_CATEGORY_XCD <- get_sum_pct(transactions_temp, "MERCHANT_CATEGORY_XCD") # 55 distincts
#MERCHANT_CATEGORY_XCD_ntile <- get_sum_pct(transactions_temp, "MERCHANT_CATEGORY_XCD_ntile") # 10 distincts
MERCHANT_COUNTRY_XCD <- get_sum_pct(transactions_temp, "MERCHANT_COUNTRY_XCD") # 123 distincts
MERCHANT_COUNTRY_XCD_ntile <- get_sum_pct(transactions_temp, "MERCHANT_COUNTRY_XCD_ntile") # 10 distincts
#MERCHANT_CITY_NAME <- get_sum_pct(transactions_temp, "MERCHANT_CITY_NAME")  # 13518 distincts. distincts
MERCHANT_CITY_NAME_ntile <- get_sum_pct(transactions_temp, "MERCHANT_CITY_NAME_ntile")  # 10 distincts
SICGROUP <- get_sum_pct(transactions_temp, "SICGROUP") # 29 distincts
TRANSACTION_CATEGORY_XCD <- get_sum_pct(transactions_temp, "TRANSACTION_CATEGORY_XCD")
TRANSACTION_TYPE_XCD <- get_sum_pct(transactions_temp, "TRANSACTION_TYPE_XCD")


pct_categ_TRANSACTION_AMT <- transactions %>% group_by(ID_CPTE, TRANSACTION_CATEGORY_XCD ) %>%
  summarise(categ_TRANSACTION_AMT = sum(TRANSACTION_AMT)) %>%
  ungroup %>%
  group_by(ID_CPTE) %>%
  mutate(pct_categ_TRANSACTION_AMT = categ_TRANSACTION_AMT / sum(categ_TRANSACTION_AMT)) %>%
  select(ID_CPTE, TRANSACTION_CATEGORY_XCD, pct_categ_TRANSACTION_AMT) %>%
  spread(key= TRANSACTION_CATEGORY_XCD, value = pct_categ_TRANSACTION_AMT, fill = 0, sep="pct_categ_TRANSACTION_AMT_")


# Ville , sicgroup et merchant_category préférée----

# 1) on sort un peu de target encoding
# 2) on sort aussi un indice de diversité
# ville préférée = "celle où tu as le plus de transaction

# on va s'en servir pour une autre variable target encodée : 
#le pourcentage de gens dans ta ville (excluant toi et quelques autres) qui a fait défaut
# ATTENTION AU LEAKAGE.. IL FAUT FAIRE UN GENRE DE OUT OF FOLD CITY DEFAULT MEAN.

# favorite_city_noQ est ta ville où tu as fait le plus de transaction durant l'année
# je vais aggréger ensuite les villes de plus petite taille sous le code 9999999
# et regarder les si les gens dans ta ville et qui ne sont pas dans ton 50-fold ont fait faillite ou pas
favorite_city_noQ <- transactions_plus %>%
  group_by(ID_CPTE)  %>%
  count(MERCHANT_CITY_NAME) %>% 
  arrange(-n) %>%
  slice(1) %>% 
  ungroup()  %>%
  rename(favorite_city_noQ= MERCHANT_CITY_NAME )%>%
  select(ID_CPTE, favorite_city_noQ)

#favorite_city %>% count(favorite_city) %>% arrange(-nn)  # il y a 1162 villes préférées..  avec 11 200 personnes c'est pas vargeux.. mais il y en a 174 avec plus de 10 personnes.. ça je suis game
favorite_city_noQ_code <- favorite_city_noQ %>% count(favorite_city_noQ) %>% arrange(-n) %>% mutate(favorite_city_noQ_code = as.factor(ifelse(n>10, favorite_city_noQ, 999999999999)))
#nlevels(favorite_city_noQ_code$favorite_city_noQ_code) #159


pouet <- performance_train %>% left_join(favorite_city_noQ) %>% left_join(favorite_city_noQ_code) %>% select(ID_CPTE, favorite_city_noQ_code, Default)
pouet$fold <- my50folds
pouet2 <- pouet %>%  group_by(favorite_city_noQ_code, fold) %>% summarise(count = n(), Default = sum(Default)) %>% ungroup


map_result <- seq_len(50) %>% purrr::map_df(~{
  
   function_result  <- list()
   message(paste0("calcul du fold ", .x))
   
   out_of_fold <- pouet2 %>% filter(fold != .x)  %>% group_by(favorite_city_noQ_code) %>% summarise(oof_mean_city_default = sum(Default) / sum(count)) %>%
     ungroup()
   
   function_result <- pouet %>% filter(fold == .x ) %>% left_join(out_of_fold)
   return(function_result)
})
oof_mean_city_default_train <- performance_train %>% left_join(map_result) %>% select(ID_CPTE, favorite_city_noQ_code,oof_mean_city_default) ## ok ca c'est pour le tarin.
#  pour ls gens test, j'utilise la population entière pour sortir de pourcentage de default
actual_mean_city_default <- performance_train %>% left_join(favorite_city_noQ) %>% left_join(favorite_city_noQ_code) %>% 
  group_by(favorite_city_noQ_code) %>% summarise( oof_mean_city_default = mean( Default))

oof_mean_city_default_test <- performance_test %>% left_join(favorite_city_noQ) %>% left_join(favorite_city_noQ_code) %>% left_join(actual_mean_city_default) %>% select(ID_CPTE, favorite_city_noQ_code, oof_mean_city_default)
 
oof_mean_city_default <- bind_rows(oof_mean_city_default_train, oof_mean_city_default_test)

## on va quand même sortir ta ville préférée dans chaque trimestre pour mettre dans le modèle

favorite_city1 <- transactions_plus %>%
  group_by(ID_CPTE,Q)  %>%
  count(MERCHANT_CITY_NAME) %>% 
  arrange(-n) %>%
  slice(1) %>% 
  ungroup()  %>%
  rename(favorite_city1= MERCHANT_CITY_NAME )%>%
  select(ID_CPTE, Q,favorite_city1)


favorite_city1_code <- favorite_city1 %>% count(favorite_city1) %>% arrange(-n) %>% mutate(favorite_city1_code = as.factor(ifelse(n>10, favorite_city1, 999999999999)))

join_this_favorite_city1_temp <- favorite_city1 %>% left_join(favorite_city1_code) %>% select(ID_CPTE, Q, favorite_city1_code)  # code de ville


join_this_favorite_city1 <- join_this_favorite_city1_temp  %>% select( ID_CPTE, Q, favorite_city1_code)%>% 
  gather(favorite_city1_code, key="key", value = "value") %>%
  unite(Var, key, Q, sep="_Q") %>%
  spread (Var, value)  %>%
  mutate(movedQ2 = as.numeric(favorite_city1_code_Q2  != favorite_city1_code_Q1),
         movedQ2 = replace_na(movedQ2, 0),
         movedQ3 = as.numeric(favorite_city1_code_Q3  != favorite_city1_code_Q2),
         movedQ3 = replace_na(movedQ3, 0),
         movedQ4 = as.numeric(favorite_city1_code_Q4  != favorite_city1_code_Q3),
         movedQ4 = replace_na(movedQ4, 0),
         moves = movedQ2 + movedQ3 + movedQ4) %>%
  mutate(favorite_city1_code_Q1 = as.factor(favorite_city1_code_Q1),  
         favorite_city1_code_Q2 = as.factor(favorite_city1_code_Q2),
         favorite_city1_code_Q3 = as.factor(favorite_city1_code_Q3),
         favorite_city1_code_Q4 = as.factor(favorite_city1_code_Q4)) 


# meme chose, on va sortir ton sicgroup préféré et ta catégorie de marchand préféré

favorite_sicgroup <- transactions %>% group_by(ID_CPTE)  %>%
  count(SICGROUP) %>% arrange(-n) %>% slice(1) %>% ungroup()  %>%
  rename(favorite_sicgroup = SICGROUP) %>%
  mutate(favorite_sicgroup = as.factor(favorite_sicgroup))  %>% 
  select(ID_CPTE,favorite_sicgroup )

join_this_favorite_sicgroup <-favorite_sicgroup 

favorite_merchant_category <- transactions %>% 
  filter( !(MERCHANT_CATEGORY_XCD %in% c("WW", "D"))) %>%
  group_by(ID_CPTE)  %>%
  count(MERCHANT_CATEGORY_XCD) %>% arrange(-n) %>% slice(1) %>% ungroup()  %>%
  rename(favorite_merchant_category = MERCHANT_CATEGORY_XCD) %>%
  mutate(favorite_merchant_category = as.factor(favorite_merchant_category))  %>% 
  select(ID_CPTE,favorite_merchant_category )

join_this_favorite_merchant_category  <- favorite_merchant_category 

## il y a2 catégories de favorite merchant qui n'existent pas dans TRAIN: WW  et D.  on va recommencer en enlevant cette possibilité..


# indice de diversité- est-ce que tu shoppes toujours dans la même ville/catégorie/pays ou pas?
join_this_index_city <- transactions %>% group_by(ID_CPTE)  %>%
  count(MERCHANT_CITY_NAME) %>%
  mutate(pct_squared =  (n /sum(n))^2) %>%
  summarise(index_city = sum(pct_squared))

join_this_index_country <- transactions %>% group_by(ID_CPTE)  %>%
  count(MERCHANT_COUNTRY_XCD) %>%
  mutate(pct_squared =  (n /sum(n))^2) %>%
  summarise(index_country = sum(pct_squared))

join_this_index_category <- transactions %>% group_by(ID_CPTE)  %>%
  count(MERCHANT_CATEGORY_XCD) %>%
  mutate(pct_squared =  (n /sum(n))^2) %>%
  summarise(index_category = sum(pct_squared))


# Temps entre les transactions ----
# on s'intéresse un peu au comportement. 
# temps moyen, minimum, ,maximum, écart type du temps entre les transactions

day_between_transaction <- transactions %>% group_by(ID_CPTE) %>%
  mutate(transaction_date = as.Date(TRANSACTION_DTTM)) %>%
  distinct(transaction_date) %>%
  arrange(transaction_date) %>%
  mutate(day_between_transaction = as.numeric(transaction_date - lag(transaction_date)))  %>%
  filter(!is.na(day_between_transaction)) %>%
  summarise(
    mean_day_between_transaction = mean(day_between_transaction),
    max_day_between_transaction = max(day_between_transaction),
    min_day_between_transaction = min(day_between_transaction),
    sum_day_between_transaction = sum(day_between_transaction),
    kurtosis_day_between_transaction = kurtosis(day_between_transaction),
    skewness_day_between_transaction = skewness(day_between_transaction),
    p25_day_between_transaction = quantile(day_between_transaction, .25),
    p75_day_between_transaction = quantile(day_between_transaction, .75),
    sd_day_between_transaction = sd(day_between_transaction)
  ) %>%
  ungroup()

# Résumé des transactions (trimestre) ----

transaction_quarter_summary <-
  transactions_plus %>% group_by(ID_CPTE, Q ) %>%
  summarise(
    MERCHANT_CITY_NAME_ndistinct = n_distinct(MERCHANT_CITY_NAME),
    MERCHANT_CATEGORY_XCD_ndistinct = n_distinct(MERCHANT_CATEGORY_XCD),
    MERCHANT_COUNTRY_XCD_ndistinct = n_distinct(MERCHANT_COUNTRY_XCD),
    TRANSACTION_TYPE_XCD_ndistinct = n_distinct(TRANSACTION_TYPE_XCD),
    SICGROUP_ndistinct = n_distinct(SICGROUP),
    count_transaction = n(),
    maxTRANSACTION_AMT = max(TRANSACTION_AMT),
    sdTRANSACTION_AMT = sd(TRANSACTION_AMT),
    p25TRANSACTION_AMT = quantile(TRANSACTION_AMT,0.25),
    p75TRANSACTION_AMT = quantile(TRANSACTION_AMT,0.75),
    meanTRANSACTION_AMT = mean(TRANSACTION_AMT),
    sumTRANSACTION_AMT = sum(TRANSACTION_AMT),
    last_transaction_date = as.Date(max(TRANSACTION_DTTM))) %>% ungroup() %>%
  gather(-ID_CPTE, -Q, key="key", value = "value") %>%
  unite(Var, key, Q, sep="_Q") %>%
  spread (Var, value)

# Résumé des transactions (annuel) ----
transaction_summary <-
  transactions %>% group_by(ID_CPTE) %>%
  summarise(
    MERCHANT_CITY_NAME_ndistinct = n_distinct(MERCHANT_CITY_NAME),
    MERCHANT_CATEGORY_XCD_ndistinct = n_distinct(MERCHANT_CATEGORY_XCD),
    MERCHANT_COUNTRY_XCD_ndistinct = n_distinct(MERCHANT_COUNTRY_XCD),
    TRANSACTION_TYPE_XCD_ndistinct = n_distinct(TRANSACTION_TYPE_XCD),
    SICGROUP_ndistinct = n_distinct(SICGROUP),
    count_transaction = n(),
    maxTRANSACTION_AMT = max(TRANSACTION_AMT),
    sdTRANSACTION_AMT = sd(TRANSACTION_AMT),
    p25TRANSACTION_AMT = quantile(TRANSACTION_AMT,0.25),
    p75TRANSACTION_AMT = quantile(TRANSACTION_AMT,0.75),
    meanTRANSACTION_AMT = mean(TRANSACTION_AMT),
    sumTRANSACTION_AMT = sum(TRANSACTION_AMT),
            last_transaction_date = as.Date(max(TRANSACTION_DTTM))) %>% ungroup() %>% 
  left_join(transaction_quarter_summary) %>%
  left_join(DECISION_XCD) %>% 
  left_join(MERCHANT_CATEGORY_XCD) %>%
  left_join(MERCHANT_COUNTRY_XCD) %>%
  left_join(MERCHANT_COUNTRY_XCD_ntile) %>%
  left_join(MERCHANT_CITY_NAME_ntile) %>%
  left_join(SICGROUP) %>%
  left_join(TRANSACTION_CATEGORY_XCD) %>%
  left_join(TRANSACTION_TYPE_XCD) %>% 
  left_join(pct_categ_TRANSACTION_AMT) %>%
  mutate(has_transaction_data = 1)  %>%
  left_join(join_this_favorite_sicgroup) %>%
  left_join(oof_mean_city_default) %>%
  left_join(join_this_favorite_city1) %>%
  left_join(join_this_favorite_merchant_category) %>%
  left_join(join_this_index_city) %>%
  left_join(join_this_index_country) %>%
  left_join(join_this_index_category) %>%
  left_join(day_between_transaction)
  
# ENFIN, on fusionne tout pour créer "mydb" ----
mydb <- performance %>% 
  left_join(paiements_summary) %>%
  left_join(transaction_summary) %>%
  left_join(facturation_summary) %>%
  mutate(year = year(PERIODID_MY)) %>%
  mutate(credit_limit_paid_per_year = sum_paiement_amt / lastCreditLimit,
         credit_limit_spent_per_year = sumTRANSACTION_AMT / lastCreditLimit,
         ratio_spent_on_paiement = sumTRANSACTION_AMT / sum_paiement_amt,
         ratio_paiement_on_owed = ifelse(lastCurrentTotalBalance>0, last_monthly_paiement_amt / lastCurrentTotalBalance, 1)) %>% # pas parfait parce que je regarde le mois au lieu de la date exacte du relevé, mais ca donne une idée
  mutate(time_since_last_paiement = PERIODID_MY - last_paiement_date ,
         lastCreditRatio_X_time_since_last_paiement = time_since_last_paiement * lastCreditRatio,
         lastCreditRatio_X_time_since_last_paiement_X_lastDelqCycle = time_since_last_paiement * lastCreditRatio* lastDelqCycle, 
         lastCashRatio_X_time_since_last_paiement = time_since_last_paiement * lastCashRatio,
         time_since_last_transaction = PERIODID_MY - last_transaction_date,
         time_between_last_transaction_and_last_paiement_date = last_transaction_date - last_paiement_date) %>%
  select(-PERIODID_MY) %>% 
  left_join(bills_outcome_summary) %>%
  mutate(count_bills  =replace_na(count_bills,0)) %>%  # si t'as pas de bills avec un dû, tu n'apparais pas dans bill_summary
  left_join(trend_delai_full_paiement) %>%
  left_join(trend_delai_next_paiement) %>%
  left_join(trend_CreditRatio) %>%
  left_join(trend_CashRatio) %>%
  left_join(pct_paid_until_next_statement_summary) %>%
  left_join(trend_pct_paid_until_next_statement) %>%
  mutate( has_transaction_data = replace_na(has_transaction_data,0),
          has_paiement_data = replace_na(has_paiement_data,0)) %>%
  select(-lastCreditRatio_ntile, -lastCreditRatio) %>%
  left_join(has_full_paiement) %>%
  mutate(has_full_paiement = replace_na(has_full_paiement,0)) %>%
  select(-delta_monthly_CurrentTotalBalance_t_minus_14)  # pas possible d'avoir une valeur pour ça
  
# Un peu de travail pour ne plus avoir de  les NA  pour permettre d'Appliquer
# autre chose que des GBM..
# TODO: mettre à jour, il reste des NA pour les variables ajoutées à la fin.
mydb <- mydb %>% mutate_if(is.Date, as.numeric) %>% mutate_if(is.difftime, as.numeric)

# identifié par has_paiement_data
paiement_vars <- c("count_paiements", "min_paiement_amt", "max_paiement_amt",
                   "sum_paiement_amt", "kurtosis_paiement_amt", "skewness_paiement_amt",
                   "count_paiement_reversal_flag", "count_monthly_paiement_amt",
                   "max_monthly_paiement_amt", "min_monthly_paiement_amt",
                   "last_monthly_paiement_amt","credit_limit_paid_per_year",
                   "ratio_spent_on_paiement", "ratio_paiement_on_owed","time_since_last_paiement",
                   "lastCreditRatio_X_time_since_last_paiement", "lastCreditRatio_X_time_since_last_paiement_X_lastDelqCycle" ,
                   "time_between_last_transaction_and_last_paiement_date", "last_paiement_date",
                   "mean_delai_next_paiement", "max_delai_next_paiement", "last_delai_next_paiement",
                   "trend_delai_next_paiement", "pct_no_next_paiement", "mean_paiement_amt","mean_monthly_paiement_amt",
                   "lastCashRatio_X_time_since_last_paiement")




# identifie par has_full_paiement
full_paiement_vars <- c("mean_delai_full_paiement", "max_delai_full_paiement", "last_delai_full_paiement",
                        "trend_delai_full_paiement" , "pct_no_full_paiement")
# identifiés par has_14_facture
t14_vars <- c("CreditRatio_t_minus_14", "delta_monthly_CurrentTotalBalance_t_minus_13",
              "CurrentTotalBalance_t_minus_14", "pct_paid_until_next_statement_t_minus_14")


# ah pis fuck j enleve les skewness et kurtosis
mydb <- mydb %>% mutate_at(vars(paiement_vars), funs(replace_na(.,0)) )  %>%
  mutate_at(vars(full_paiement_vars), funs(replace_na(.,0))) %>%
  mutate(has_14_facture = as.numeric(count_facture == 14)) %>%
  mutate_at(vars(t14_vars), funs(replace_na(.,0)) ) %>%
  mutate(has_0_bill = as.numeric(count_bills == 0))  %>%
  select(-starts_with("kurtosis"), -starts_with("skewness"))

mydb %>% select_if(function(x) any(is.na(x))) %>% 
  summarise_all(funs(sum(is.na(.)))) -> NA_mydb

NA_mydb %>% glimpse

table(sapply(mydb, class)) # lister le nombre de variable de chaque classe

mydb <- mydb %>% select(-mean_Default_lastCreditRatio_ntile)

write_rds(mydb, path="mydb.rds")


#mydb <- read_rds(path="mydb.rds")

label_var <- "Default"
feature_vars <- mydb %>% select(-Default, -ID_CPTE) %>% colnames
factor_vars <-  mydb %>% select_if(is.factor) %>% colnames
myformula <- paste0(label_var,  "~", paste0( factor_vars, collapse = " + ") ) %>% as.formula()

mydb_dummy <-    caret::dummyVars(myformula, data=mydb  , fullRank = TRUE) %>%
   predict(newdata = mydb) %>%
   as_tibble() %>% bind_cols(mydb %>% select(-one_of(factor_vars) ))#- ,ID_CPTE
mydb_dummy<- mydb_dummy %>% select(-ID_CPTE)
write_rds(mydb_dummy, "mydb_dummy.rds")


# Un premier boost, pour rire ----

feature_vars_dummy <-  
  colnames(mydb_dummy %>% 
             select(-Default ))


# on doit se faire un train test sinon xgboost va overfitter.
# create 20% test data
# split the remaining 80% into 72% train and 8% watchlist
alltrain_dummy <- mydb_dummy  %>% filter(!is.na(Default))
write_rds(alltrain_dummy, "alltrain_dummy.rds")

trainIndex2 <- caret::createDataPartition(
  alltrain_dummy %>% pull(label_var), 
  p = .90, list = FALSE, times = 1)

train_dummy <-alltrain_dummy[trainIndex2, ]
wlist_dummy <-alltrain_dummy[-trainIndex2, ]
test_dummy <- mydb_dummy %>% filter(is.na(Default))
write_rds(test_dummy, "test_dummy.rds")

# ensuite on converti en DMatrix, l'input requis par xgboost (noter l'absence de l'offet, on le rajoute juste après)
alltrain_xgbmatrix <- xgb.DMatrix(
  data = alltrain_dummy %>% select(feature_vars_dummy) %>% data.matrix() ,  # évite erreur amalgamation?
  label = alltrain_dummy %>% pull(label_var),
  missing = "NAN")


train_xgbmatrix <- xgb.DMatrix(
  data = train_dummy %>% select(feature_vars_dummy)  %>% data.matrix(), 
  label = train_dummy %>% pull(label_var),
  missing = "NAN")

wlist_xgbmatrix <- xgb.DMatrix(
  data = wlist_dummy %>% select(feature_vars_dummy) %>% data.matrix(), 
  label = wlist_dummy %>% pull(label_var),
  missing = "NAN")

test_xgbmatrix <- xgb.DMatrix(
  data = test_dummy %>% select(feature_vars_dummy)  %>% data.matrix(), 
  label = test_dummy %>% pull(label_var),
  missing = "NAN")


myWatch <- list(wlist=wlist_xgbmatrix, 
                train=train_xgbmatrix)

myParam <- list(
  booster = "gbtree",

  eta = .02,
  gamma = 0.00,
  max.depth = 4,
  min_child_weight=3,
  subsample = 0.50,
  colsample_bytree = 0.5,
  objective = 'binary:logistic' ,
  eval_metric = "auc")

#callback is used to tell the model to stop creating trees when the watchlist stops getting any better. not available in caret.
myCallback <- list(
  # cb.early.stop(
  #   metric_name = "wlist_auc", 
  #   stopping_rounds = 100,
  #   maximize= TRUE),
  cb.print.evaluation(period = 50))


start <- Sys.time()
booster <- xgb.train(
  params = myParam, 
  data = train_xgbmatrix, 
  nround = 50,
  watchlist=myWatch,
  callbacks = myCallback)
duree <- Sys.time() - start
duree 

# table - variable importance
var_importance <- xgb.importance(
  feature_names = feature_vars_dummy,
  model = booster) %>% tbl_df()

#Il utilise environ 300 variables sur les 2800+ générées
write_csv(var_importance, "var_importance.csv")

