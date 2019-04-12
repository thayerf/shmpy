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
## transformation for interval responses
logit = function(x) log(x / (1 - x))
## create the response variables
y = params
interval_params = 5:9
y[,interval_params] = logit(y[,interval_params])
y = as.matrix(y)
y = scale(y)

## check how some of the models do at predicting
recurrent_model_1 = load_model_hdf5("recurrent_model_1.hdf5")
input_data_recurrent = one_hot_2d_sequences(seqs)
recurrent_model_1_preds = predict(recurrent_model_1, x = input_data_recurrent)
for(i in 1:9) {
    cat(sprintf("Recurrent 1, correlation for %s: %f", colnames(y)[i], cor(recurrent_model_1_preds[,i], y[,i])), "\n")
}

recurrent_model_2 = load_model_hdf5("recurrent_model_2.hdf5")
recurrent_model_2_preds = predict(recurrent_model_2, x = input_data_recurrent)
for(i in 1:9) {
    cat(sprintf("Recurrent 2, correlation for %s: %f", colnames(y)[i], cor(recurrent_model_2_preds[,i], y[,i])), "\n")
}


conv_model_1 = load_model_hdf5("convolutional_model_1.hdf5")
input_data_convolutional = one_hot_2d_sequences_for_conv_net(seqs)
conv_model_1_preds = predict(conv_model_1, x = input_data_convolutional)
for(i in 1:9) {
    cat(sprintf("Convolutional 1, correlation for %s: %f", colnames(y)[i], cor(conv_model_1_preds[,i], y[,i])), "\n")
}

conv_model_2 = load_model_hdf5("convolutional_model_2.hdf5")
conv_model_2_preds = predict(conv_model_2, x = input_data_convolutional)
for(i in 1:9) {
    cat(sprintf("Convolutional 2, correlation for %s: %f", colnames(y)[i], cor(conv_model_2_preds[,i], y[,i])), "\n")
}

dense_model = load_model_hdf5("logit_ber_model.hdf5")
input_data_dense = one_hot_1d_sequences(seqs)
dense_model_preds = predict(dense_model, x = input_data_dense)
for(i in 1:9) {
    cat(sprintf("Dense, correlation for %s: %f", colnames(y)[i], cor(dense_model_preds[,i], y[,i])), "\n")
}
