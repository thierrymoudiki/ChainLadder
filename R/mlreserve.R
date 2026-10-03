###################################################################
##  mlReserve: machine-learning-based insurance loss reserving
##
##  Generalisation of ChainLadder::glmReserve (Wayne Zhang) in the
##  spirit of Techtonique/mlS3 and Techtonique/unifiedml: the GLM is
##  replaced by *any* regression model exposed through
##    - a matrix interface   fit_func(x, y, ...)        (glmnet, randomForest, ranger, svm, xgboost, ...)
##    - a formula interface  fit_func(formula, data, ...) (lm, glm, gam, ranger, rpart, ...)
##    - caret                fit_func = "<caret method>"  (passed to caret::train)
##    - an R6 object with $fit(X, y, ...) / $predict(X)  (unifiedml::Model$new(...))
##
##  Because generic ML models have no vcov(), prediction errors are
##  obtained by a (Pearson- or log-) residual bootstrap, optionally using
##  out-of-fold (cross-validated) residuals, which are far more honest
##  than in-sample residuals for flexible learners.
##
##  Arguments
##    triangle     a ChainLadder 'triangle' (exposure attribute honoured:
##                 the learner models value / exposure)
##    fit_func     learner: function, caret method name, or R6 learner
##    predict_func optional function(model, newX) -> numeric; by default
##                 the model's predict method is inspected for its
##                 new-data argument (newx / newdata / data)
##    interface    "auto" detects from fit_func (first formal 'formula'
##                 -> formula, character -> caret, R6 -> R6, else matrix)
##    features     "factor"  one-hot origin + dev (chain-ladder structure)
##                 "numeric" origin, dev, calendar period as numbers
##                 "both", or function(lda) -> data.frame for custom ones
##    transform    "none" (model incrementals), "log" or "log1p" (model
##                 log incrementals, Duan smearing for the mean; log1p
##                 tolerates zero incrementals); "asinh" (signed log,
##                 tolerates zeros *and* negative incrementals/recoveries)
##    mse.method   "bootstrap" or "none"
##    resid.type   "cv" (out-of-fold, default) or "insample" residuals
##    resid.scale  "pearson" (resid / sqrt(mu), ODP-like) or "raw"
##    hetero       scale residuals by development period (recommended)
##    positive     redraw negative bootstrap pseudo-responses (needed for
##                 learners such as Poisson/Tweedie GLMs)
##    ...          passed to fit_func (or caret::train, or $fit)
###################################################################

mlReserve <- function(triangle,
                      fit_func,
                      predict_func = NULL,
                      interface = c("auto", "matrix", "formula", "caret", "R6"),
                      features = c("factor", "numeric", "both"),
                      transform = c("none", "log", "log1p", "asinh"),
                      cum = TRUE,
                      mse.method = c("bootstrap", "none"),
                      resid.type = c("cv", "insample"),
                      resid.scale = c("pearson", "raw"),
                      nfolds = 5L,
                      nsim = 250L,
                      hetero = TRUE,
                      positive = FALSE,
                      seed = NULL,
                      verbose = FALSE,
                      ...) {
  call <- match.call()
  interface   <- match.arg(interface)
  transform   <- match.arg(transform)
  mse.method  <- match.arg(mse.method)
  resid.type  <- match.arg(resid.type)
  resid.scale <- match.arg(resid.scale)
  if (!is.function(features)) features <- match.arg(features)
  if (!is.null(seed)) set.seed(seed)

  if (!inherits(triangle, "triangle"))
    stop("triangle must be of class 'triangle'")

  ## ---------------------------------------------------------------
  ## 1. data: incremental triangle -> long format
  ## ---------------------------------------------------------------
  tr.incr <- if (cum) ChainLadder::cum2incr(triangle) else triangle
  lda <- as.data.frame(tr.incr, origin = names(dimnames(tr.incr))[1],
                       dev = names(dimnames(tr.incr))[2])
  names(lda)[1:3] <- c("origin", "dev", "value")
  lda$origin <- factor(lda$origin, levels = dimnames(triangle)[[1]])
  lda$dev_f  <- factor(lda$dev, levels = sort(unique(lda$dev)))
  lda$exposure <- .expand_exposure(attr(tr.incr, "exposure"), lda)

  ldaFit <- lda[!is.na(lda$value), , drop = FALSE]
  ldaOut <- lda[ is.na(lda$value), , drop = FALSE]
  if (nrow(ldaOut) == 0) stop("triangle has no cells to project")
  ldaOut$origin_out <- droplevels(ldaOut$origin)

  ## response on the modelling scale: z = g(value / exposure)
  yrate <- ldaFit$value / ldaFit$exposure
  if (transform == "log") {
    if (any(yrate <= 0))
      stop("transform = 'log' needs strictly positive incremental values (try 'log1p')")
    z <- log(yrate)
  } else if (transform == "log1p") {
    if (any(yrate <= -1)) stop("transform = 'log1p' needs incremental values > -1")
    z <- log1p(yrate)
  } else if (transform == "asinh") {
    z <- asinh(yrate)                        # signed log: handles zeros and recoveries
  } else z <- yrate

  ## ---------------------------------------------------------------
  ## 2. learner engine (unified fit / predict)
  ## ---------------------------------------------------------------
  if (interface == "auto") interface <- .detect_interface(fit_func)
  engine <- .make_engine(fit_func, predict_func, interface, list(...))
  feat <- function(d) .make_features(d, features, interface)

  Xfit <- feat(ldaFit)
  Xout <- feat(ldaOut)

  model <- engine$fit(Xfit, z)
  zfit  <- engine$predict(model, Xfit)
  zout  <- engine$predict(model, Xout)

  ## ---------------------------------------------------------------
  ## 3. residuals (modelling scale), Pearson-type scaling
  ## ---------------------------------------------------------------
  scl <- function(mu) {
    if (transform != "none" || resid.scale == "raw") rep(1, length(mu))
    else sqrt(pmax(abs(mu), 1e-8 * max(abs(z))))
  }
  if (resid.type == "cv") {
    zcv <- .cv_predict(engine, feat, ldaFit, z, nfolds)
    res <- (z - zcv) / scl(zfit)
  } else {
    res <- (z - zfit) / scl(zfit)
  }
  ## heteroscedasticity adjustment: residual scale per development period
  ## (pooled with neighbouring periods until >= 5 residuals), as in
  ## England & Verrall's hetero-adjusted ODP bootstrap. Without it, the
  ## small, noisy tail cells dominate the residual pool.
  dfit <- as.integer(ldaFit$dev_f); dout <- as.integer(ldaOut$dev_f)
  nd <- nlevels(lda$dev_f)
  ok <- is.finite(res)
  sdd <- rep(1, nd)
  if (hetero) {
    for (d in seq_len(nd)) {
      w <- 0
      repeat {
        sel <- ok & abs(dfit - d) <= w
        if (sum(sel) >= 5 || w >= nd) break
        w <- w + 1
      }
      sdd[d] <- sqrt(mean(res[sel]^2))
    }
    sdd[!is.finite(sdd) | sdd <= 0] <- sqrt(mean(res[ok]^2))
  }
  res_ok <- res[ok] / sdd[dfit[ok]]          # standardised residuals
  res_ok <- res_ok - mean(res_ok)            # centred for the bootstrap
  # Duan smearing per development period so that exp(log-mean) is the mean
  smear  <- if (transform != "none") vapply(sdd, function(s) mean(exp( s * res_ok)), 1) else rep(1, nd)
  smearm <- if (transform != "none") vapply(sdd, function(s) mean(exp(-s * res_ok)), 1) else rep(1, nd)

  back <- function(zz, d)                    # modelling scale -> mean incremental
    switch(transform, none = zz, log = exp(zz) * smear[d], log1p = exp(zz) * smear[d] - 1,
           asinh = (exp(zz) * smear[d] - exp(-zz) * smearm[d]) / 2)   # E[sinh(z + e)]

  ## ---------------------------------------------------------------
  ## 4. point estimate of reserves
  ## ---------------------------------------------------------------
  yp <- back(zout, dout) * ldaOut$exposure
  resMeanAy  <- tapply(yp, ldaOut$origin_out, sum)
  resMeanTot <- sum(resMeanAy)

  ## ---------------------------------------------------------------
  ## 5. bootstrap prediction errors
  ## ---------------------------------------------------------------
  resMeanAyB <- resPredAyB <- matrix(0)
  S.E <- rep(NA_real_, length(resMeanAy) + 1)
  if (mse.method == "bootstrap") {
    nO <- length(z); nA <- length(resMeanAy)
    resMeanAyB <- resPredAyB <- matrix(NA_real_, nsim, nA)
    sfit <- scl(zfit) * sdd[dfit]
    for (i in seq_len(nsim)) {
      zB <- zfit + sample(res_ok, nO, replace = TRUE) * sfit
      if (positive) {                       # redraw negative pseudo-responses
        for (k in 1:100) {
          neg <- zB < 0
          if (!any(neg)) break
          zB[neg] <- zfit[neg] + sample(res_ok, sum(neg), replace = TRUE) * sfit[neg]
        }
      }
      zoutB <- tryCatch(engine$predict(engine$fit(Xfit, zB), Xout),
                        error = function(e) { if (verbose) message(conditionMessage(e)); NULL })
      if (is.null(zoutB)) next
      ymB <- back(zoutB, dout)
      # process noise: draw a (rescaled) residual for every future cell
      eps <- sample(res_ok, length(zoutB), replace = TRUE) * sdd[dout]
      ypB <- switch(transform, none = zoutB + eps * scl(zoutB),
                    log = exp(zoutB + eps), log1p = expm1(zoutB + eps),
                    asinh = sinh(zoutB + eps))
      resMeanAyB[i, ] <- tapply(ymB * ldaOut$exposure, ldaOut$origin_out, sum)
      resPredAyB[i, ] <- tapply(ypB * ldaOut$exposure, ldaOut$origin_out, sum)
      if (verbose && i %% 50 == 0) message("bootstrap ", i, "/", nsim)
    }
    keep <- stats::complete.cases(resPredAyB)
    if (sum(keep) < nsim) warning(nsim - sum(keep), " bootstrap refits failed and were dropped")
    resMeanAyB <- resMeanAyB[keep, , drop = FALSE]
    resPredAyB <- resPredAyB[keep, , drop = FALSE]
    colnames(resMeanAyB) <- colnames(resPredAyB) <- levels(ldaOut$origin_out)
    S.E <- c(apply(resPredAyB, 2, stats::sd), stats::sd(rowSums(resPredAyB)))
  }

  ## ---------------------------------------------------------------
  ## 6. assemble results (same layout as glmReserve)
  ## ---------------------------------------------------------------
  IBNR <- round(c(resMeanAy, total = resMeanTot))
  Latest <- ChainLadder::getLatestCumulative(ChainLadder::incr2cum(tr.incr))
  Latest <- Latest[names(Latest) %in% levels(ldaOut$origin_out)]
  Latest <- c(Latest, total = sum(Latest))
  Ultimate <- Latest + IBNR
  resDf <- data.frame(Latest = Latest, Dev.To.Date = Latest / Ultimate,
                      Ultimate = Ultimate, IBNR = IBNR,
                      S.E = S.E, CV = S.E / IBNR)
  row.names(resDf) <- names(Latest)

  full <- rbind(ldaFit[, c("origin", "dev", "value")],
                transform(ldaOut[, c("origin", "dev")], value = round(yp)))
  FullTriangle <- ChainLadder::as.triangle(full, origin = "origin", dev = "dev", value = "value")
  if (cum) FullTriangle <- ChainLadder::incr2cum(FullTriangle)

  out <- list(call = call, summary = resDf,
              Triangle = triangle, FullTriangle = FullTriangle,
              model = model, interface = interface,
              fitted = back(zfit, dfit) * ldaFit$exposure,
              predicted = data.frame(origin = ldaOut$origin, dev = ldaOut$dev, value = yp),
              residuals = res,
              sims.reserve.mean = resMeanAyB,
              sims.reserve.pred = resPredAyB)
  class(out) <- "mlReserve"
  out
}

###################################################################
## helpers
###################################################################

.expand_exposure <- function(expo, lda) {
  if (is.null(expo)) return(rep(1, nrow(lda)))
  if (!is.null(names(expo)) && all(as.character(lda$origin) %in% names(expo)))
    return(as.numeric(expo[as.character(lda$origin)]))
  numorig <- suppressWarnings(as.numeric(as.character(lda$origin)))
  if (anyNA(numorig))
    stop("Unnamed exposures need origin values convertible to numeric.")
  as.numeric(expo[numorig - min(numorig) + 1])
}

.is_R6_learner <- function(f)
  is.environment(f) && is.function(f$fit) && is.function(f$predict)

.detect_interface <- function(fit_func) {
  if (is.character(fit_func)) return("caret")
  if (.is_R6_learner(fit_func)) return("R6")
  if (!is.function(fit_func)) stop("fit_func must be a function, a caret method name or an R6 learner")
  fm <- names(formals(fit_func))
  if (length(fm) && fm[1] %in% c("formula", "form")) "formula" else "matrix"
}

## features: chain-ladder-like one-hot (origin + dev), numeric trends
## (origin, dev, calendar), both, or a user function(lda) -> data.frame
.make_features <- function(d, features, interface) {
  if (is.function(features)) {
    df <- as.data.frame(features(d))
  } else {
    num <- data.frame(origin_num = as.numeric(d$origin),
                      dev_num    = as.numeric(d$dev_f))
    num$cal_num <- num$origin_num + num$dev_num - 1
    fac <- data.frame(origin = d$origin, dev = d$dev_f)
    df <- switch(features, factor = fac, numeric = num, both = cbind(fac, num))
  }
  if (interface == "formula") return(df)
  # matrix-type learners: full one-hot coding of factors (no reference level)
  facs <- names(df)[vapply(df, is.factor, logical(1))]
  ctr <- lapply(df[facs], function(f) stats::contrasts(f, contrasts = FALSE))
  X <- stats::model.matrix(~ ., data = df, contrasts.arg = if (length(facs)) ctr else NULL)
  X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
  colnames(X) <- make.names(colnames(X))
  if (interface == "caret") as.data.frame(X) else X
}

.as_pred_vector <- function(p) {
  if (is.list(p) && !is.null(p$predictions)) p <- p$predictions    # ranger
  if (is.list(p) && !is.null(p$pred)) p <- p$pred
  if (is.matrix(p) || is.data.frame(p)) p <- as.matrix(p)[, ncol(p)] # e.g. glmnet path -> last lambda
  as.numeric(p)
}

.make_engine <- function(fit_func, predict_func, interface, dots) {
  ## default predictor: look up the predict method's signature to find how
  ## new data is passed (newx = glmnet, newdata = most, data = ranger, ...)
  default_pred <- function(m, X) {
    meth <- NULL
    for (cl in class(m)) {
      meth <- utils::getS3method("predict", cl, optional = TRUE)
      if (!is.null(meth)) break
    }
    args <- list(m)
    fm <- if (is.null(meth)) character(0) else names(formals(meth))
    newarg <- intersect(c("newx", "newdata", "data", "new_data", "x"), fm)[1]
    if (is.na(newarg)) args <- c(args, list(X)) else args[[newarg]] <- X
    if (inherits(m, c("glm", "gam", "glmnet")) && "type" %in% fm) args$type <- "response"
    do.call(stats::predict, args)
  }
  pred <- function(m, X)
    .as_pred_vector(if (is.null(predict_func)) default_pred(m, X) else predict_func(m, X))

  switch(interface,
    matrix = list(
      fit = function(X, y) do.call(fit_func, c(list(X, y), dots)),
      predict = pred),
    formula = list(
      fit = function(X, y) do.call(fit_func, c(list(.value ~ ., data = cbind(.value = y, X)), dots)),
      predict = pred),
    caret = {
      if (!requireNamespace("caret", quietly = TRUE)) stop("package 'caret' is required")
      if (is.null(dots$trControl)) dots$trControl <- caret::trainControl(method = "none")
      list(
        fit = function(X, y) do.call(caret::train, c(list(x = X, y = y, method = fit_func), dots)),
        predict = pred)
    },
    R6 = list(
      fit = function(X, y) {
        obj <- if (is.function(fit_func$clone_model)) fit_func$clone_model() else fit_func$clone(deep = TRUE)
        do.call(obj$fit, c(list(X, y), dots))
        obj
      },
      predict = function(m, X) .as_pred_vector(if (is.null(predict_func)) m$predict(X) else predict_func(m, X)))
  )
}

## out-of-fold predictions; a cell is only scored if its origin and its
## development period each keep >= 2 cells in the training fold (otherwise
## the held-out residual measures non-identifiability, not prediction error)
.cv_predict <- function(engine, feat, ldaFit, z, nfolds) {
  n <- length(z)
  folds <- sample(rep_len(seq_len(nfolds), n))
  zcv <- rep(NA_real_, n)
  for (k in seq_len(nfolds)) {
    te <- folds == k; tr <- !te
    no <- table(ldaFit$origin[tr]); nd <- table(ldaFit$dev_f[tr])
    ok <- te & no[as.character(ldaFit$origin)] >= 2 & nd[as.character(ldaFit$dev_f)] >= 2
    if (!any(ok)) next
    zcv[ok] <- tryCatch(engine$predict(engine$fit(feat(ldaFit[tr, ]), z[tr]), feat(ldaFit[ok, ])),
                        error = function(e) NA_real_)
  }
  zcv
}

###################################################################
## S3 methods
###################################################################

summary.mlReserve <- function(object, type = c("triangle", "model"), ...) {
  type <- match.arg(type)
  if (type == "triangle") object$summary else summary(object$model)
}

print.mlReserve <- function(x, ...) {
  cat("mlReserve (", x$interface, " interface)\n\n", sep = "")
  print(x$summary)
  invisible(x)
}

residuals.mlReserve <- function(object, ...) object$residuals
fitted.mlReserve <- function(object, ...) object$fitted

quantile.mlReserve <- function(x, probs = c(0.75, 0.95, 0.995), ...) {
  sim <- x$sims.reserve.pred
  if (ncol(sim) < 2 && all(sim == 0)) stop("no simulations: use mse.method = 'bootstrap'")
  sim <- cbind(sim, total = rowSums(sim))
  t(apply(sim, 2, stats::quantile, probs = probs, ...))
}

# 1 original triangle, 2 full triangle, 3 predictive distribution,
# 4 residuals vs fitted, 5 normal QQ of residuals
plot.mlReserve <- function(x, which = 1, ...) {
  if (which == 1) plot(x$Triangle, ...)
  else if (which == 2) plot(x$FullTriangle, ...)
  else if (which == 3) {
    sim <- x$sims.reserve.pred
    if (ncol(sim) < 2 && all(sim == 0)) stop("no simulations: use mse.method = 'bootstrap'")
    sim <- cbind(sim, total = rowSums(sim))
    nm <- colnames(sim)
    df <- data.frame(group = factor(rep(nm, each = nrow(sim)), levels = nm),
                     prediction = as.vector(sim))
    print(ggplot2::ggplot(df, ggplot2::aes(.data$prediction)) +
            ggplot2::geom_density() +
            ggplot2::facet_wrap(~group, nrow = 2, scales = "free") +
            ggplot2::scale_x_continuous("predicted reserve"))
  } else if (which == 4) {
    plot(x$fitted, x$residuals, xlab = "fitted", ylab = "residual", ...)
    abline(h = 0, col = "#B3B3B3")
  } else if (which == 5) {
    r <- x$residuals[is.finite(x$residuals)]
    qqnorm(r, ...); qqline(r)
  }
}
