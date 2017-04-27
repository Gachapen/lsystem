require(ggplot2)
require(gridExtra)

stats <- read.csv(file = "~/daicloud/school/15HMACSA/imt4904-master_thesis/data/learning-stats-fitness-reward-1.3-1.csv", header = TRUE)

plot_stat <- function(stat, label) {
        ggplot(stats, aes(sample, stat)) +
                geom_point(size = 1, alpha = 1 / 5) +
                geom_smooth() +
                geom_smooth(method = lm, color = "red") +
                ylab(label)
}

plot_all <- function() {
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
