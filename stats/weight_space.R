library(ggplot2)
library(rgl)

plot_weight_space_1 <- function(file) {
  samples <- read.csv(file = file, header = FALSE)
  ggplot() + geom_point(data = samples, aes(V2, 0))
}

plot_weight_space_2 <- function(file) {
  samples <- read.csv(file = file, header = FALSE)
  ggplot() + geom_point(data = samples, aes(V2, V3))
}

plot_weight_space_3 <- function(file) {
  samples <- read.csv(file = file, header = FALSE)
  plot3d(samples$V2, samples$V3, samples$V4);
}
