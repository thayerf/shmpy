library(keras)
library(Matrix)
library(shmr)
## read in the data
params = read.csv("for_nnet_params.csv", header = FALSE)
colnames(params) = c("ber_lambda",
            "bubble_size",
            "exo_left",
            "exo_right",
            "ber_params[0]",
            "ber_params[1]",
            "ber_params[2]",
            "ber_params[3]",
            "p_fw")
seqs = read.csv("for_nnet_sequences.csv", header = FALSE, stringsAsFactors = FALSE)[,1]
train_indices = 1:(.9 * length(seqs))
## transformation for interval responses
logit = function(x) log(x / (1 - x))
## create the response variables
y = params
interval_params = 5:9
y[,interval_params] = logit(y[,interval_params])
y = as.matrix(y)
y = scale(y)
## dense model
predictors_dense = one_hot_1d_sequences(seqs)
model_dense = keras_model_sequential()
model_dense %>%
#    layer_dense(units = 512, activation = 'relu', input_shape = ncol(predictors_dense)) %>%
#    layer_dropout(rate = .2) %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dropout(rate = .2) %>%
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dropout(rate = .2) %>%
    layer_dense(units = ncol(y))
model_dense %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(lr = .001),
  metrics = c('accuracy')
)
weights = fit_and_save_weights(model_dense, epochs = 25,
    x = predictors_dense[train_indices,], y = y[train_indices,],
    batch_size = 128, validation_split = .2)
## compare to the same model fit with a large step size
model_dense_large_step = keras_model_sequential()
model_dense_large_step %>%
#    layer_dense(units = 512, activation = 'relu', input_shape = ncol(predictors_dense)) %>%
#    layer_dropout(rate = .2) %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dropout(rate = .2) %>%
    layer_dense(units = 100, activation = 'relu') %>%
    layer_dropout(rate = .2) %>%
    layer_dense(units = ncol(y))
model_dense_large_step %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(lr=.003),
  metrics = c('accuracy')
)
weights_large_step = fit_and_save_weights(model_dense_large_step, epochs = 25,
    x = predictors_dense[train_indices,], y = y[train_indices,],
    batch_size = 128, validation_split = .2)

large_step_compare = avg_weight_prediction_comparison(model_dense_large_step, weights_large_step,
    x = predictors_dense[-train_indices,], burnin = 10)
## MSE for final weights
mean((large_step_compare[[1]] - y[-train_indices,])^2)
## MSE for average weights
mean((large_step_compare[[2]] - y[-train_indices,])^2)

compare = avg_weight_prediction_comparison(model_dense, weights,
    x = predictors_dense[-train_indices,], burnin = 10)
## MSE for final weights
mean((compare[[1]] - y[-train_indices,])^2)
## MSE for average weights
mean((compare[[2]] - y[-train_indices,])^2)


model_dense %>% fit(
  predictors_dense, y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2
)
save_model_hdf5(model_dense, filepath = "dense_model.hdf5")

cat("convolutional model")
## convolutional model
input_data_convolutional = one_hot_2d_sequences_for_conv_net(seqs)
input_shape = list(dim(input_data_convolutional)[2], dim(input_data_convolutional)[3], 1)
latent_dim = 20
model_conv_1 = keras_model_sequential()
model_conv_1 %>%
    layer_conv_2d(filters = latent_dim, kernel_size = c(7, 4), input_shape = input_shape) %>%
    layer_flatten() %>%
    layer_dropout(rate=.2) %>%
#    layer_dense(units = 1500, activation = 'relu') %>%
#    layer_dropout(rate=.2) %>%
    layer_dense(units = 20, activation = 'relu') %>%
    layer_dropout(rate=.2) %>%
    layer_dense(units = ncol(y))
model_conv_1 %>% compile(loss = 'mse', optimizer = optimizer_rmsprop())
model_conv_1 %>% fit(
 input_data_convolutional, y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2
)
save_model_hdf5(model_conv_1, filepath = "convolutional_model_1.hdf5")

cat("convolutional model 2")
## convolutional model 2
input_shape = list(dim(input_data_convolutional)[2], dim(input_data_convolutional)[3], 1)
model_conv_2 = keras_model_sequential()
model_conv_2 %>%
    layer_conv_2d(filters = latent_dim, kernel_size = c(7, 4), input_shape = input_shape) %>%
    layer_flatten() %>%
    layer_dropout(rate=.2) %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dropout(rate=.2) %>%
    layer_dense(units = ncol(y))
model_conv_2 %>% compile(loss = 'mse', optimizer = optimizer_rmsprop())
model_conv_2 %>% fit(
  input_data_convolutional, y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2
)
save_model_hdf5(model_conv_2, filepath = "convolutional_model_2.hdf5")

cat("recurrent model 1")
## recurrent model 1
input_data_recurrent = one_hot_2d_sequences(seqs)
model_recurrent_1 = keras_model_sequential()
model_recurrent_1 %>%
    layer_lstm(units = latent_dim, return_state = FALSE, return_sequences = TRUE) %>%
    layer_flatten() %>%
    layer_dense(units = 200, activation = 'relu') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = ncol(y))
model_recurrent_1 %>% compile(loss = 'mse', optimizer = optimizer_rmsprop())
model_recurrent_1 %>% fit(
  input_data_recurrent, y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2
)
save_model_hdf5(model_recurrent_1, filepath = "recurrent_model_1.hdf5")

cat("recurrent model 2")
## recurrent model 2
model_recurrent_2 = keras_model_sequential()
model_recurrent_2 %>%
    layer_lstm(units = latent_dim, return_state = FALSE, return_sequences = FALSE) %>%
    layer_dense(units = 20, activation = 'relu') %>%
    layer_dense(units = ncol(y))
model_recurrent_2 %>% compile(loss = 'mse', optimizer = optimizer_rmsprop())
model_recurrent_2 %>% fit(
  input_data_recurrent, y, 
  epochs = 50, batch_size = 128, 
  validation_split = 0.2
)
save_model_hdf5(model_recurrent_2, filepath = "recurrent_model_2.hdf5")
