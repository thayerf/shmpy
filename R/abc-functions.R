#' Main function for fitting model by ABC
#'
#' @param sim_params Parameters used in the simulator.
#' @param sim_statistics Statistics corresponding to each parameter set.
#' @param obs_statistics Statistics for the observed data.
#' @param match_tol What fraction of the samples to keep.
#' @export
#' @return A matrix, columns representing parameters, of accepted parameters.
abc_rej <- function(sim_params, sim_statistics, obs_statistics, match_tol = .01) {
    obs_sim_dists = apply(sim_statistics, 1, function(x) sum((x - obs_statistics)^2))
    threshold = quantile(obs_sim_dists, probs = match_tol)
    return(sim_params[which(obs_sim_dists <= threshold),])
}



#' Performs ABC on a simulated data set
#'
#' @param params A matrix, nsims x nparams, giving the parameter
#' values used in the simulations.
#' @param statistics A matrix, nsims x nstats, giving the summary
#' statistics computed from each of the simulations.
#' @param match_tol Keep the closest match_tol fraction of the
#' parameters for the posterior.
#'
#' @return A data frame with columns for the true value of the
#' parameter, the posterior mean of the parameter, and the name of the
#' parameter.
#' @export
check_abc_on_sims <- function(params, statistics, match_tol) {
    get_param_comparisons <- function(idx, params, statistics, match_tol) {
        true_param = params[idx,]
        non_rej = abc_rej(sim_params = params[-idx,],
            sim_statistics = statistics[-idx,],
            obs_statistics = statistics[idx,],
            match_tol = match_tol)
        posterior_means = colMeans(non_rej)
        parameter_comparisons = data.frame(
            true_param = unlist(true_param),
            posterior_means = unlist(posterior_means),
            parameter = colnames(non_rej))
        return(parameter_comparisons)
    }

    comparisons = lapply(
        1:nrow(params),
        get_param_comparisons,
        params = params,
        statistics = statistics,
        match_tol = match_tol)
    comparisons = Reduce(rbind, comparisons)
    return(comparisons)
}

#' Creates summary statistics
#'
#' Given a set of sequences, a function that takes a set of sequences
#' and returns a predictor matrix, and a net, computes the net's
#' predictions and returns the average of the predictions over the
#' entire set of sequences.
#'
#' @param seqs A character vector, length equal to the number of sequences.
#' @param predictor_creation_fn A function that takes a sequence
#' vector and returns a predictor matrix.
#' @param net A trained net.
#' @param groups A factor describing how the sequences should be
#' aggregated. If groups of sequences were drawn from the same
#' parameters, the summary statistics are averaged over each group.
#' @param extra_preds Any non-sequence predictors to use.
#'
#' @return A matrix of summary statistics, number of rows equal to the
#' number of groups, number of columns equal to the number of summary
#' statistics (the number of parameters the net estimates).
#' @export
get_net_summary_stats <- function(seqs, predictor_creation_fn, net, groups, extra_preds = NULL) {
    predictors = predictor_creation_fn(seqs)
    if(!is.null(extra_preds)) {
        predictors = cbind(predictors, extra_preds)
    }
    predictions = predict(net, predictors)
    split_predictions = lapply(unique(groups), function(g) predictions[which(groups == g),])
    avg_predictions = lapply(split_predictions, colMeans)
    summary_stats = Reduce(rbind, avg_predictions)
    rownames(summary_stats) = NULL
    return(summary_stats)
}
