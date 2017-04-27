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
    geom_point(data=stats, aes(sample, current), size = 1, alpha = 1 / 5, color="black") +
    geom_smooth(data=stats, aes(sample, current), color="black") +
    geom_point(data=stats, aes(sample, local.mean), size = 1, alpha = 1 / 5, color="blue") +
    geom_smooth(data=stats, aes(sample, local.mean), color="blue") +
    geom_point(data=stats, aes(sample, local.best), size = 1, alpha = 1 / 5, color="red") +
    geom_smooth(data=stats, aes(sample, local.best), color="red")
}

plot_all <- function(file) {
  stats <- read.csv(file = file, header = TRUE)

  grid.arrange(
        plot_stat(stats$current, "score"),
        plot_stat(stats$mean, "mean"),
        plot_stat(stats$variance, "variance"),
        plot_stat(stats$best, "best"),
        plot_stat(stats$local.mean, "local mean"),
        plot_stat(stats$local.variance, "local variance"),
        plot_stat(stats$local.best, "local best"),
        nrow = 3,
        ncol = 3)
}
