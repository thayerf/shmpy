## Using the pinball loss to estimate posterior quantiles



```r
library(keras)
```


```r
## model: x ~ N(mu, 1)
## we want to know mu

## we need to define a new loss function, the pinball loss
tilted_loss = function(q, y, f) {
  e = y - f
  k_mean(k_maximum(q * e, (q - 1) * e), axis = 2)
}
```

We draw values of $\mu$ from the prior and then draw $x$ given $\mu$.

```r
## prior on mu is N(0, prior_sd)
## draw prior, data poirs
prior_sd = 5
prior_draws = rnorm(n = 1000, mean = 0, sd = prior_sd)
data_draws = sapply(prior_draws, function(mu) rnorm(n = 1, mean = mu, sd = 1))
```

Set up the net to estimate the posterior quantiles:

```r
## we want to estimate the .05, .5, and .95 quantile of the posterior
quantiles = c(.05, .5, .95)
quantile_losses = lapply(quantiles, function(q) {
    function(y_true, y_pred) tilted_loss(q, y_true, y_pred)
})

## define the net
input = layer_input(shape=1,name="input")
base_model = input  %>%
  layer_dense(units = 10, activation='relu') 
q1 = base_model %>% 
  layer_dense(units = 1, name="q1") 
q2 = base_model %>% 
  layer_dense(units = 1, name="q2") 
q3 = base_model %>% 
  layer_dense(units = 1, name="q3")
model = keras_model(input,list(q1,q2,q3)) %>%
  compile(optimizer = "adam",
          loss=quantile_losses,
          metrics="mae")
## fit on our data
model %>% fit(x = matrix(data_draws, ncol = 1),
           y = list(prior_draws, prior_draws, prior_draws),
           epochs = 20)
```


Then do the same thing assuming that we have more data points: draw $\mu$ from the prior, draw $x_1,\ldots, x_{n_\text{samples}}$ from $\mu$.


```r
n_samples = 100
prior_draws2 = rnorm(n = 10000, mean = 0, sd = prior_sd)
data_draws2 = t(sapply(prior_draws2, function(mu) rnorm(n = n_samples, mean = mu, sd = 1)))
```

We make a net that has three output units (for the three quantiles we want to estimate), and predicts $\mu$ from a set of $n_{\text{samples}}$ data values:

```r
## define the net
input2 = layer_input(shape=n_samples,name="input2")
base_model2 = input2  %>%
    layer_dense(units = 10, activation='relu')
q12 = base_model2 %>% 
  layer_dense(units = 1, name="q12") 
q22 = base_model2 %>% 
  layer_dense(units = 1, name="q22") 
q32 = base_model2 %>% 
  layer_dense(units = 1, name="q32")

model2 = keras_model(input2,list(q12,q22,q32)) %>%
  compile(optimizer = "adam",
          loss=quantile_losses,
          metrics="mae")
model2 %>% fit(x = data_draws2,
               y = list(prior_draws2, prior_draws2, prior_draws2),
               epochs = 20)
```

Let's check against what we know the posterior should be:

```r
## true posterior is N(x_obs * prior_sd^2 / (1 + prior_sd^2), (1 + 1 / prior_sd^2)^(-1))
x_obs = -1
posterior_mean = x_obs * prior_sd^2 / (1 + prior_sd^2)
posterior_variance = (1 + 1 / prior_sd^2)^(-1)
qnorm(p = quantiles, mean = posterior_mean, sd = sqrt(posterior_variance))
```

```
## [1] -2.5744501 -0.9615385  0.6513732
```

```r
## our estimates of the quantiles
model %>% predict(x = x_obs)
```

```
## [[1]]
##          [,1]
## [1,] -2.11439
## 
## [[2]]
##           [,1]
## [1,] -1.160386
## 
## [[3]]
##            [,1]
## [1,] -0.2450408
```

```r
## true posterior is N(xbar_obs * prior_sd^2 / (1/n + prior_sd^2), (1 / prior_sd^2 + n)^(-1))
x_obs = -1
posterior_mean = x_obs * prior_sd^2 / (1 / n_samples + prior_sd^2)
posterior_variance = (1 / prior_sd^2 + n_samples)^(-1)
qnorm(p = quantiles, mean = posterior_mean, sd = sqrt(posterior_variance))
```

```
## [1] -1.1640526 -0.9996002 -0.8351477
```

```r
## our estimates of the quantiles
model2 %>% predict(x = matrix(x_obs, nrow = 1, ncol = n_samples))
```

```
## [[1]]
##           [,1]
## [1,] -1.244066
## 
## [[2]]
##          [,1]
## [1,] -1.00724
## 
## [[3]]
##            [,1]
## [1,] -0.8356974
```
