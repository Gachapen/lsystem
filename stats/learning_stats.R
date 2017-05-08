require(ggplot2)
require(gridExtra)

stats <- read.csv(file = "learning-stats.csv", header = TRUE)

plot_stat <- function(stat, label) {
  ggplot(stats, aes(sample, stat)) +
    geom_point(size = 1, alpha = 1 / 5) +
    geom_smooth() +
    geom_smooth(method = lm, color = "red") +
    ylab(label)
}

plot_combined <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    # ylim(c(-3.1, 0.0)) +
    # xlim(c(0, 1024)) +
    # geom_point(data = stats, aes(sample, current), size = 1, alpha = 1 / 5, color = "black") +
    geom_smooth(data = stats, aes(sample, current), color = "black") +
    # geom_smooth(data = stats, aes(sample, current), method = lm, color = "black") +
    # geom_point(data = stats, aes(sample, local.mean), size = 1, alpha = 1 / 5, color="blue") +
    geom_smooth(data = stats, aes(sample, local.mean), color = "blue") +
    # geom_smooth(data = stats, aes(sample, local.mean), method = lm, color = "blue") +
    # geom_point(data = stats, aes(sample, local.best), size = 1, alpha = 1 / 5, color="red") +
    geom_smooth(data = stats, aes(sample, local.best), color = "red")
    # geom_smooth(data = stats, aes(sample, local.best), method = lm, color = "red")
}

plot_all <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  grid.arrange(
    plot_stat(stats$current, "score"),
    # plot_stat(stats$mean, "mean"),
    # plot_stat(stats$variance, "variance"),
    # plot_stat(stats$best, "best"),
    plot_stat(stats$local.mean, "local mean"),
    plot_stat(stats$local.variance, "local variance"),
    plot_stat(stats$local.best, "local best"),
    nrow = 2,
    ncol = 2)
}

plot_new <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    geom_point(data = stats, aes(samples, score), size = 1, color = "black") +
    geom_smooth(data = stats, aes(samples, score), color = "blue") +
    geom_smooth(data = stats, aes(samples, score), method = lm, color = "red")
}

plot_sa <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  ggplot() +
    scale_x_continuous(breaks = seq(0, nrow(stats), by = 64)) +
    geom_point(data = stats[stats$accepted == "true",], aes(iteration, score), size = 1, color = "blue") +
    geom_smooth(data = stats, aes(iteration, score), color = "blue") +
    geom_point(data = stats[stats$accepted == "false",], aes(iteration, score), size = 1, color = "red") +
    geom_smooth(data = stats[stats$accepted == "false",], aes(iteration, score), color = "red")
}
