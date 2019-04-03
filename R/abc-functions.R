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
