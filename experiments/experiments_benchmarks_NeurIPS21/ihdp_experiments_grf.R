library(grf)
library(reticulate)

do_ihdp_exper <- function(n_exp = 100,
                          n_reps = 5,
                          setup = 'original') {
  # read IHDP data (originally saved in numpy format)
  np <- import("numpy")
  npz_train <- np$load('catenets/datasets/data/ihdp_npci_1-100.train.npz')

  x_train <- npz_train$f[['x']]
  y_train <- npz_train$f[['yf']]
  w_train <- npz_train$f[['t']]
  mu0_train <- npz_train$f[['mu0']]
  mu1_train <- npz_train$f[['mu1']]


  npz_test <- np$load('catenets/datasets/data/ihdp_npci_1-100.test.npz')

  x_test <- npz_test$f[['x']]
  y_test <- npz_test$f[['yf']]
  w_test <- npz_test$f[['t']]
  mu0_test <- npz_test$f[['mu0']]
  mu1_test <- npz_test$f[['mu1']]


  if (setup == 'modified') {
    # make TE additive instead
    y_train[w_train == 1] = y_train[w_train == 1] + mu0_train[w_train == 1]
    mu1_train = mu0_train + mu1_train
    mu1_test = mu0_test + mu1_test
  }

  cate_train <- mu1_train - mu0_train
  cate_test <- mu1_test - mu0_test

  for (i in 1:n_exp) {
    # loop over runs
    print(paste0('Experiment number', i))
    for (k in 1:n_reps) {
      # loop over seeds

      # Causal forest ------------------------------
      print('causal forest')
      cf <-
        causal_forest(x_train[, , i], y_train[, i], w_train[, i], seed = k)

      # predict CATE
      pred_cf_in <- predict(cf, x_train[, , i])$predictions
      pred_cf_out <- predict(cf, x_test[, , i])$predictions

      # Evaluate
      rmse_cf_in <- sqrt(mean((cate_train[, i] - pred_cf_in) ^ 2))
      rmse_cf_out <- sqrt(mean((cate_test[, i] - pred_cf_out) ^ 2))


      # T-learner -----------------------------------------------------
      print('t learner')
      y0.forest <- regression_forest(subset(x_train[, , i], w_train[, i] == 0),
                                     y_train[w_train[, i] == 0, i], seed = k)
      y1.forest <- regression_forest(subset(x_train[, , i], w_train[, i] == 1),
                                     y_train[w_train[, i] == 1, i], seed = k)
      # predict CATE
      pred_t_in <-
        predict(y1.forest, x_train[, , i])$predictions - predict(y0.forest, x_train[, , i])$predictions
      pred_t_out <-
        predict(y1.forest, x_test[, , i])$predictions - predict(y0.forest, x_test[, , i])$predictions
      # Evaluate
      rmse_t_in <- sqrt(mean((cate_train[, i] - pred_t_in) ^ 2))
      rmse_t_out <- sqrt(mean((cate_test[, i] - pred_t_out) ^ 2))

      # s-learner -------------------------------------------------------------
      print('s learner')
      s_forest <-
        regression_forest(cbind(x_train[, , i], w_train[, i]), y_train[, i], seed =
                            k)
      # create extended feature matrices
      n_train <- nrow(x_train[, , i])
      n_test <- nrow(x_test[, , i])
      train_treated <- rep(1, n_train)
      train_control <- rep(0, n_train)
      test_treated <- rep(1, n_test)
      test_control <- rep(0, n_test)

      # predict CATE
      pred_s_in <-
        predict(s_forest, cbind(x_train[, , i], train_treated))$predictions - predict(s_forest, cbind(x_train[, , i], train_control))$predictions
      pred_s_out <-
        predict(s_forest, cbind(x_test[, , i], test_treated))$predictions - predict(s_forest, cbind(x_test[, , i], test_control))$predictions
      # evaluate
      rmse_s_in <- sqrt(mean((cate_train[, i] - pred_s_in) ^ 2))
      rmse_s_out <- sqrt(mean((cate_test[, i] - pred_s_out) ^ 2))


      df_res <-
        data.frame(
          simu = i,
          run = k,
          cf_in = rmse_cf_in,
          t_in = rmse_t_in,
          s_in = rmse_s_in,
          cf_out = rmse_cf_out,
          t_out = rmse_t_out,
          s_out = rmse_s_out
        )


      if (i * k == 1) {
        write.table(
          df_res,
          file = paste0('results/experiments_benchmarking/ihdp/grf_', setup, '.csv'),
          col.names = T,
          sep = ',',
          row.names = F
        )
      }
      else{
        write.table(
          df_res,
          file = paste0('results/experiments_benchmarking/ihdp/grf_', setup, '.csv'),
          col.names = F,
          append = T,
          sep = ',',
          row.names = F
        )
      }
    }
  }
}
