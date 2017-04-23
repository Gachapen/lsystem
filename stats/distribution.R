require(ggplot2)
require(gridExtra)

distribution <- read.csv(file = "distribution.csv", header = TRUE)

productions <- function() {
  prods <- distribution[distribution$rule == "productions",]
  ggplot(prods, aes(alternative + 1, weight)) +
    geom_col() +
    xlab("num productions") +
    ggtitle("production count distribution")
}

operation <- function() {
  operations <- distribution[distribution$rule == "operation",]
  ggplot(operations, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="operation")) +
    scale_fill_discrete(labels=c("+", "-", "^", "&", ">", "<")) +
    ggtitle("operation distribution")
}

strlen <- function() {
  strlens <- distribution[distribution$rule == "string" & distribution$choice == 0,]
  ggplot(strlens, aes(depth, weight, fill=factor(alternative + 1))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="string length", ncol=2)) +
    ggtitle("string length distribution")
}

stack <- function() {
  stacks <- distribution[distribution$rule == "string" & distribution$choice == 1,]
  ggplot(stacks, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="symbol vs stack")) +
    scale_fill_discrete(labels=c("symbol", "stack")) +
    ggtitle("symbol/stack distribution")
}

symbol <- function() {
  symbols <- distribution[distribution$rule == "symbol",]
  ggplot(symbols, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="symbol type")) +
    scale_fill_discrete(labels=c("variable", "operation")) +
    ggtitle("symbol distribution")
}

variable <- function() {
  variables <- distribution[distribution$rule == "variable",]
  ggplot(variables, aes(depth, weight, fill=factor(alternative))) +
    geom_col(position="dodge") +
    guides(fill=guide_legend(title="variable", ncol=2)) +
    scale_fill_discrete(labels=LETTERS) +
    ggtitle("variable distribution")
}

plot_all <- function() {
  grid.arrange(
    productions(),
    strlen(),
    stack(),
    symbol(),
    variable(),
    operation(),
    nrow=3,
    ncol=2)
}
