# Run PKNCA and NonCompart on the scenarios of the benchmark datasets.
#
# Every scenario of docs/data/benchmarks/scenarios.csv is analysed with PKNCA
# and with NonCompart under the settings Phoenix WinNonlin used for the
# published reference output; the results are written to
# docs/data/benchmarks/pknca/<scenario>.csv (the long table of
# `as.data.frame(pk.nca(...))`) and docs/data/benchmarks/noncompart/<scenario>.csv
# (the wide table of `tblNCA`), which docs/benchmark_datasets.md compares
# pkpdutils against. The versions of R and of both packages are written to
# docs/data/benchmarks/versions.csv. Run from the root of the repository:
#
#   Rscript scripts/benchmark_datasets.R

suppressPackageStartupMessages({
  library(PKNCA)
  library(NonCompart)
})

dir <- file.path("docs", "data", "benchmarks")
scenarios <- read.csv(file.path(dir, "scenarios.csv"), stringsAsFactors = FALSE)
dir.create(file.path(dir, "pknca"), showWarnings = FALSE)
dir.create(file.path(dir, "noncompart"), showWarnings = FALSE)

# the observed peak, which PKNCA computes on its own interval without
# imputation: an imputed value at the dose would otherwise enter it (the
# back-extrapolated C0 of a bolus is larger than every sample)
observed <- c("cmax", "tmax", "cmax.dn")
# the parameters PKNCA computes for every scenario, plus the ones of the route
common <- c(
  "tlast", "clast.obs", "clast.pred", "auclast", "aucall",
  "aucinf.obs", "aucinf.pred", "aumclast", "aumcinf.obs", "aumcinf.pred",
  "aucpext.obs", "aucpext.pred", "half.life", "lambda.z", "r.squared",
  "adj.r.squared", "lambda.z.time.first", "lambda.z.time.last",
  "lambda.z.n.points", "span.ratio", "cl.obs", "cl.pred", "vz.obs", "vz.pred",
  "aucinf.obs.dn", "aucinf.pred.dn"
)
by_route <- list(
  oral = c("tlag", "mrt.last", "mrt.obs", "mrt.pred"),
  iv_bolus = c("c0", "mrt.iv.last", "mrt.iv.obs", "mrt.iv.pred", "vss.iv.obs", "vss.iv.pred"),
  iv_infusion = c("mrt.iv.last", "mrt.iv.obs", "mrt.iv.pred", "vss.iv.obs", "vss.iv.pred")
)
# The value at the dose of a curve whose first sample comes later: 0 after an
# extravascular dose and an infusion and the log-linear back extrapolation of
# the first two samples after a bolus, PKNCA's `pk.calc.c0(method =
# "logslope")`. PKNCA looks an imputation up by the name
# `PKNCA_impute_method_<name>`; both are defined here, since PKNCA 0.12.1 has
# no back extrapolation and its `start_conc0` also overwrites a value measured
# at the dose with 0 (the predose samples of the theophylline study), where
# Phoenix WinNonlin inserts a value only when there is none.
PKNCA_impute_method_start_zero <- function(conc, time, start = 0, ..., options = list()) {
  ret <- data.frame(conc = conc, time = time)
  if (!any(time %in% start)) {
    ret <- rbind(ret, data.frame(time = start, conc = 0))
    ret <- ret[order(ret$time), ]
  }
  ret
}
PKNCA_impute_method_start_logslope <- function(conc, time, start = 0, ..., options = list()) {
  ret <- data.frame(conc = conc, time = time)
  if (!any(time %in% start)) {
    c0 <- pk.calc.c0(conc, time, time.dose = start, method = "logslope")
    ret <- rbind(ret, data.frame(time = start, conc = c0))
    ret <- ret[order(ret$time), ]
  }
  ret
}
impute <- c(oral = "start_zero", iv_bolus = "start_logslope", iv_infusion = "start_zero")
# the administration and the trapezoidal rule as NonCompart names them
adm <- c(oral = "Extravascular", iv_bolus = "Bolus", iv_infusion = "Infusion")
down <- c(linear = "Linear", linear_log = "Log")
pknca_method <- c(linear = "linear", linear_log = "lin up/log down")

for (i in seq_len(nrow(scenarios))) {
  s <- scenarios[i, ]
  data <- read.csv(file.path(dir, s$dataset))
  data <- data.frame(
    Subject = data[[s$subject]], time = data[[s$time]], conc = data[[s$conc]]
  )
  duration <- if (is.na(s$duration)) 0 else s$duration
  exclude_cmax <- tolower(s$exclude_cmax) == "true"

  # PKNCA: one interval from the dose to infinity
  doses <- data.frame(Subject = unique(data$Subject), time = 0, dose = s$dose, duration = duration)
  o_conc <- PKNCAconc(data, conc ~ time | Subject)
  o_dose <- PKNCAdose(
    doses, dose ~ time | Subject,
    route = if (s$route == "oral") "extravascular" else "intravascular",
    duration = "duration"
  )
  parameters <- c(observed, common, by_route[[s$route]])
  intervals <- data.frame(start = 0, end = Inf, impute = c(NA, impute[[s$route]]))
  intervals[parameters] <- FALSE
  intervals[1, observed] <- TRUE
  intervals[2, c(common, by_route[[s$route]])] <- TRUE
  o_data <- PKNCAdata(
    o_conc, o_dose,
    intervals = intervals,
    impute = "impute",
    options = list(
      auc.method = pknca_method[[s$auc_method]],
      min.hl.points = 3,
      adj.r.squared.factor = 1e-4,
      allow.tmax.in.half.life = !exclude_cmax
    )
  )
  pknca <- as.data.frame(pk.nca(o_data))
  pknca <- pknca[, c("Subject", "start", "end", "PPTESTCD", "PPORRES", "exclude")]
  write.csv(pknca, file.path(dir, "pknca", paste0(s$scenario, ".csv")), row.names = FALSE)

  # NonCompart: the call of the validation report of NonCompart against WinNonlin
  noncompart <- tblNCA(
    data, "Subject", "time", "conc",
    dose = s$dose, adm = adm[[s$route]], dur = duration, down = down[[s$auc_method]],
    R2ADJ = if (s$dataset == "indometh.csv") 0.8 else 0.7,
    doseUnit = s$dose_unit, timeUnit = s$time_unit, concUnit = s$unit
  )
  noncompart <- noncompart[, colnames(noncompart) != "b0"]
  write.csv(noncompart, file.path(dir, "noncompart", paste0(s$scenario, ".csv")), row.names = FALSE)
}

versions <- data.frame(
  tool = c("R", "PKNCA", "NonCompart"),
  version = c(
    paste(R.version$major, R.version$minor, sep = "."),
    as.character(packageVersion("PKNCA")),
    as.character(packageVersion("NonCompart"))
  )
)
write.csv(versions, file.path(dir, "versions.csv"), row.names = FALSE)
