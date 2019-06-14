#' Fit a net and save the weights at each epoch
#'
#' @param compiled_model A compiled model
#' @param epochs Number of training epochs.
#' @param ... Other parameters to be passed to the `fit` function.
#' @return A list containing the set of weights.
fit_and_save_weights <- function(compiled_model, epochs, ...) {
    weight_list = list()
    for(i in 1:epochs) {
        fit(compiled_model, ..., epochs = 1)
        weight_list[[i]] = get_weights(compiled_model)
    }
    return(weight_list)
}

#' Predicts on averaged weights and final weights
#'
#' @param compiled_model
#' @param weight_list
#' @param burning
#' @param ... Other arguments to be passed to `predict`.
#'
#' @return A list, first element containing the predictions using the final weights, the second element containing the predictions using the average weights.
avg_weight_prediction_comparison <- function(compiled_model, weight_list,
                                             burnin = .1 * length(weight_list), ...) {
    preds_final_weights = predict(compiled_model, ...)
    avg_weights = average_weights(weight_list, burnin)
    set_weights(compiled_model, avg_weights)
    preds_avg_weights = predict(compiled_model, ...)
    # return the weights in the model to the initial weights
    set_weights(compiled_model, weight_list[[length(weight_list)]])
    return(list(preds_final_weights, preds_avg_weights))
}

#' Averages weights over several epochs
average_weights <- function(weight_list, burnin) {
    ## the weights for each epoch are given as a list
    ## across epochs, we should have weights for each layer as vectors or matrices of the same size
    n_layers = length(weight_list[[1]])
    avg_weights = list()
    ## remove the burnin
    weight_list = weight_list[(burnin + 1):length(weight_list)]
    for(i in 1:n_layers) {
        layer_weight_list = lapply(weight_list, function(x) x[[i]])
        a = array(unlist(layer_weight_list), dim = c(dim(layer_weight_list[[1]]), length(layer_weight_list)))
        avg_weights[[i]] = apply(a, MARGIN = 1:(length(dim(a)) - 1), FUN = mean)
    }
    return(avg_weights)
}
