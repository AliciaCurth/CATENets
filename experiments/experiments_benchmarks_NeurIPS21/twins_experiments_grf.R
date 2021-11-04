library(grf)

do_twins_exper <- function(
                          n_reps = 10,
                          subset_train = 500,
                          test_size = 0.5,
                          treat_prop=0.5) {
  i=1
  for (k in 0:(n_reps-1)) {
      # loop over seeds
      print(paste0('Iteration number ', k))
      set.seed(k)
      
      # read data (need to run the catenets script first; that creates the preprocessed data)
      if (subset_train == 5700){
        df_train <- read.csv(paste0('experiments/experiments_benchmarks_NeurIPS21/twins_datasets/preprocessed_', treat_prop, '_None_', test_size, '_', k, '_train.csv'))
        df_test <- read.csv(paste0('experiments/experiments_benchmarks_NeurIPS21/twins_datasets/preprocessed_', treat_prop, '_None_', test_size, '_', k, '_test.csv'))
      }else{
      df_train <- read.csv(paste0('experiments/experiments_benchmarks_NeurIPS21/twins_datasets/preprocessed_', treat_prop, '_', subset_train, '_', test_size, '_', k, '_train.csv'))
      df_test <- read.csv(paste0('experiments/experiments_benchmarks_NeurIPS21/twins_datasets/preprocessed_', treat_prop, '_', subset_train, '_', test_size, '_', k, '_test.csv'))
      }
      X_train <- data.matrix(df_train[,2:40])
      X_test <- data.matrix(df_test[,2:40])
      
      
      z_train = df_train$w
      y_train = df_train$y
      t_train = df_train$y1 - df_train$y0
      
      
      z_test = df_test$w
      y_test = df_test$y
      t_test = df_test$y1 - df_test$y0
      
      # causal forest
      print('causal forest')
      cf <- causal_forest(X_train, y_train, z_train, seed = k)
      pred_cf_in <- predict(cf, X_train)$predictions
      pred_cf_out <- predict(cf, X_test)$predictions
      rmse_cf_in <- sqrt(mean((t_train - pred_cf_in) ^ 2))
      rmse_cf_out <- sqrt(mean((t_test - pred_cf_out) ^ 2))

      
      # t-learner
      print('t learner')
      y0.forest <-
        regression_forest(subset(X_train, z_train == 0), y_train[z_train == 0],  seed =
                            k * i)
      y1.forest <-
        regression_forest(subset(X_train, z_train == 1), y_train[z_train == 1],  seed =
                            k * i)
      pred_t_in <-
        predict(y1.forest, X_train)$predictions - predict(y0.forest, X_train)$predictions
      pred_t_out <-
        predict(y1.forest, X_test)$predictions - predict(y0.forest, X_test)$predictions
      rmse_t_in <- sqrt(mean((t_train - pred_t_in) ^ 2))
      rmse_t_out <- sqrt(mean((t_test - pred_t_out) ^ 2))
    
      
      # s-learner
      print('s learner')
      s_forest <-
        regression_forest(cbind(X_train, z_train), y_train,  seed = k * i)
      n_train <- nrow(X_train)
      n_test <- nrow(X_test)
      train_treated <- rep(1, n_train)
      train_control <- rep(0, n_train)
      test_treated <- rep(1, n_test)
      test_control <- rep(0, n_test)
      pred_s_in <-
        predict(s_forest, cbind(X_train, train_treated))$predictions - predict(s_forest, cbind(X_train, train_control))$predictions
      
      pred_s_out <-
        predict(s_forest, cbind(X_test, test_treated))$predictions - predict(s_forest, cbind(X_test, test_control))$predictions
      rmse_s_in <- sqrt(mean((t_train - pred_s_in) ^ 2))
      rmse_s_out <- sqrt(mean((t_test - pred_s_out) ^ 2))
      
      
    
      df_res <-
          data.frame(
            run = k,
            cf_in = rmse_cf_in,
            t_in = rmse_t_in,
            s_in = rmse_s_in,
            cf_out = rmse_cf_out,
            t_out = rmse_t_out,
            s_out = rmse_s_out
          )
      
      
      if (k == 0) {
        write.table(
          df_res,
          file = paste0(
            'results/experiments_benchmarking/twins/twins_grf_',
            subset_train,
            '_',
            n_reps,
            '.csv'
          ),
          col.names = T,
          sep = ',',
          row.names = F
        )
      }
      else{
        write.table(
          df_res,
          file = paste0(
            'results/experiments_benchmarking/twins/twins_grf_',
            subset_train,
            '_',
            n_reps,
            '.csv'
          ),
          col.names = F,
          sep = ',',
          row.names = F,
          append = T
        )
      }
      
  }
}
