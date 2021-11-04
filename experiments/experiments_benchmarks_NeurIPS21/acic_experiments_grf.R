library(grf)

do_acic_exper_loop <-
  function(simnums = c(2, 26, 7),
           n_reps = 5,
           n_exp = 10,
           with_t = F) {
    # function to loop over multiple simulation settings
    for (k in simnums) {
      do_acic_exper(k,
                    n_reps = n_reps,
                    n_exp = n_exp,
                    with_t = with_t)
    }
  }

do_acic_exper <- function(simnum,
                          n_reps = 5,
                          n_exp = 10,
                          with_t = F) {
  # function to do acic experiments for one simulation setting (simnum)
  # n_reps indicates the number of replications (random seeds used)
  # n_exp indicates the number of simulations to use within this setting (1-100)
  # with_t indicates whether to create additional results with pre-transformed data
  
  X <- data.matrix(read.csv('catenets/datasets/data/data_cf_all/x.csv'))
  X_trans <- data.matrix(read.csv('catenets/datasets/data/x_trans.csv'))
  range_train = 1:4000
  range_test = 4001:4802
  
  # get files
  sim_dir = paste0('catenets/datasets/data/data_cf_all/', simnum, '/')
  file_list <- list.files(sim_dir)
  
  for (i in 1:(n_exp)) {
    # loop over simulations within this setting
    print(paste0('Experiment number ', i))
    for (k in 1:n_reps) {
      # loop over seeds
      print(paste0('Iteration number ', k))
      set.seed(k * i)
      
      X_train <- X[range_train,]
      X_test <- X[range_test,]
      X_t_train <- X_trans[range_train,]
      
      outcomes = read.csv(paste0(sim_dir, file_list[i]))
      z = outcomes$z
      y = outcomes$z * outcomes$y1 + (1 - outcomes$z) * outcomes$y0
      t = outcomes$mu1 - outcomes$mu0
      
      z_train = z[range_train]
      y_train = y[range_train]
      t_train = t[range_train]
      t_test = t[range_test]
      
      # causal forest
      print('causal forest')
      cf <- causal_forest(X_train, y_train, z_train, seed = k * i)
      pred_cf <- predict(cf, X)$predictions
      rmse_cf_in <- sqrt(mean((t_train - pred_cf[range_train]) ^ 2))
      rmse_cf_out <- sqrt(mean((t_test - pred_cf[range_test]) ^ 2))
      
      if (with_t == T) {
        # also fit estimators using pre-transformed data
        cf.t <- causal_forest(X_t_train, y_train, z_train,  seed = k * i)
        pred_cf.t <- predict(cf.t, X_trans)$predictions
        rmse_cf_in.t <- sqrt(mean((t_train - pred_cf.t[range_train]) ^ 2))
        rmse_cf_out.t <- sqrt(mean((t_test - pred_cf.t[range_test]) ^ 2))
      }
      
      # t-learner
      print('t learner')
      y0.forest <-
        regression_forest(subset(X_train, z_train == 0), y_train[z_train == 0],  seed =
                            k * i)
      y1.forest <-
        regression_forest(subset(X_train, z_train == 1), y_train[z_train == 1],  seed =
                            k * i)
      pred_t <-
        predict(y1.forest, X)$predictions - predict(y0.forest, X)$predictions
      rmse_t_in <- sqrt(mean((t_train - pred_t[range_train]) ^ 2))
      rmse_t_out <- sqrt(mean((t_test - pred_t[range_test]) ^ 2))
      
      if (with_t == T) {
        # also fit estimators using pre-transformed data
        y0.forest.t <-
          regression_forest(subset(X_t_train, z_train == 0), y_train[z_train == 0],  seed =
                              k * i)
        y1.forest.t <-
          regression_forest(subset(X_t_train, z_train == 1), y_train[z_train == 1],  seed =
                              k * i)
        pred_t.t <-
          predict(y1.forest.t, X_trans)$predictions - predict(y0.forest.t, X_trans)$predictions
        rmse_t_in.t <- sqrt(mean((t_train - pred_t.t[range_train]) ^ 2))
        rmse_t_out.t <- sqrt(mean((t_test - pred_t.t[range_test]) ^ 2))
      }
      
      # s-learner
      print('s learner')
      s_forest <-
        regression_forest(cbind(X_train, z_train), y_train,  seed = k * i)
      n_total <- nrow(X)
      test_treated <- rep(1, n_total)
      test_control <- rep(0, n_total)
      pred_s <-
        predict(s_forest, cbind(X, test_treated))$predictions - predict(s_forest, cbind(X, test_control))$predictions
      rmse_s_in <- sqrt(mean((t_train - pred_s[range_train]) ^ 2))
      rmse_s_out <- sqrt(mean((t_test - pred_s[range_test]) ^ 2))
      
      if (with_t == T) {
        # also fit estimators using pre-transformed data
        s_forest.t <-
          regression_forest(data.matrix(cbind(X_t_train, z_train)), y_train,  seed =
                              k * i)
        pred_s.t <-
          predict(s_forest.t, data.matrix(cbind(X_trans, test_treated)))$predictions - predict(s_forest.t, data.matrix(cbind(X_trans, test_control)))$predictions
        rmse_s_in.t <- sqrt(mean((t_train - pred_s.t[range_train]) ^ 2))
        rmse_s_out.t <- sqrt(mean((t_test - pred_s.t[range_test]) ^ 2))
      }
      
      
      if (with_t == T) {
        df_res <-
          data.frame(
            file = file_list[i],
            run = k,
            cf_in = rmse_cf_in,
            cf_t_in = rmse_cf_in.t,
            t_in = rmse_t_in,
            t_t_in = rmse_t_in.t,
            s_in = rmse_s_in,
            s_in_t = rmse_s_in.t,
            cf_out = rmse_cf_out,
            cf_t_out = rmse_cf_out.t,
            t_out = rmse_t_out,
            t_t_out = rmse_t_out,
            s_out = rmse_s_out,
            s_t_out = rmse_s_out.t
          )
      } else{
        df_res <-
          data.frame(
            file = file_list[i],
            run = k,
            cf_in = rmse_cf_in,
            t_in = rmse_t_in,
            s_in = rmse_s_in,
            cf_out = rmse_cf_out,
            t_out = rmse_t_out,
            s_out = rmse_s_out
          )
      }
      
      if (i * k == 1) {
        write.table(
          df_res,
          file = paste0(
            'results/experiments_benchmarking/acic2016/grf_',
            simnum,
            '_',
            with_t,
            '_',
            n_exp,
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
            'results/experiments_benchmarking/acic2016/grf_',
            simnum,
            '_',
            with_t,
            '_',
            n_exp,
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
}
