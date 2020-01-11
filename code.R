#========================================
# This script is part of the capstone project of HarvardX
# Professional Certificate in Data Science.
#
# In this project, we apply the concepts learned during the
# course, such as data wrangling, data exploration and
# visualization, statistics and probability, R and R Markdown,
# and machine learning.
#
# The goal of this project is to predict wine type, red or white,
# and the quality of red wine from the physicochemical properties.
#
# The dataset comes from UCI Machine Learning repository at
# https://archive.ics.uci.edu/ml/datasets/Wine+Quality
#
#------------------------------------------------
# Suggestion: run the script in chunks, so you can gradually see the
# results. If you run all at once, you won't understand what's
# being done.
#
#------------------------------------------------
# Dataset information provided by the the authors:
#
# The two datasets are related to red and white variants of 
# the Portuguese "Vinho Verde" wine.
# For more details, consult: http://www.vinhoverde.pt/en/ or 
# the reference [Cortez et al., 2009].
# Due to privacy and logistic issues, only physicochemical
# (inputs) and sensory (the output) variables are available 
# (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks.
# The classes are ordered and not balanced (e.g. there are munch 
# more normal wines than excellent or poor ones). Outlier detection 
# algorithms could be used to detect the few excellent
# or poor wines. Also, we are not sure if all input variables are relevant. So
# it could be interesting to test feature selection methods. 
# 
# Number of Instances: red wine - 1599; white wine - 4898. 
# 
# Number of Attributes: 11 + output attribute
# 
# Note: several of the attributes may be correlated, thus it makes
# sense to apply some sort of feature selection.
# 
# Attribute information:
#   
# For more information, read [Cortez et al., 2009].
# 
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
# Output variable (based on sensory data): 
#   12 - quality (score between 0 and 10)
#========================================

# Create the F1 score plot, used to explain this metric.

library(tidyverse)

# Define F1 score function
f1_score <- function(prec, rec) {
  2 * (prec * rec) / (prec + rec)
}

# Create a dataframe with precision and recall values
m <- expand.grid(precision = seq(0, 1, .01), 
                 recall = seq(0, 1, .01))

tbb <- tibble(precision = m$precision, 
              recall = m$recall, 
              F1 = f1_score(m$precision, m$recall))

# Create the plot
tbb %>% ggplot(aes(precision, recall, z = F1, fill = F1)) +
  geom_raster() +
  labs(title = "F1 Score") +
  xlab("Precision") +
  ylab("Recall") +
  scale_fill_gradientn(colors=c("#F70D0D", "white", "#005DFF")) +
  # Draw countour lines
  stat_contour(breaks=c(0.1), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.2), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.3), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.4), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.5), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.6), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.7), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.8), color="black", na.rm = TRUE) +
  stat_contour(breaks=c(0.9), color="black", na.rm = TRUE) +
  # Write the line levels
  geom_text(aes(x = 0.15, y = 0.1, label = "0.1")) +
  geom_text(aes(x = 0.25, y = 0.2, label = "0.2")) +
  geom_text(aes(x = 0.35, y = 0.3, label = "0.3")) +
  geom_text(aes(x = 0.45, y = 0.4, label = "0.4")) +
  geom_text(aes(x = 0.55, y = 0.5, label = "0.5")) +
  geom_text(aes(x = 0.65, y = 0.6, label = "0.6")) +
  geom_text(aes(x = 0.75, y = 0.7, label = "0.7")) +
  geom_text(aes(x = 0.85, y = 0.8, label = "0.8")) +
  geom_text(aes(x = 0.95, y = 0.9, label = "0.9")) 


#========================================
# Prepare the dataset
#========================================
# This section downloas the files from UCI, import in R
# and creates the training and testing sets.

# Set number of significant digits
options(digits = 3)

#---------------------------------------
# Install and load the liberaries used in this section
#---------------------------------------
# The 'load_lib' function installs and loads
# a vector of libraries
load_lib <- function(libs) {
  sapply(libs, function(lib) {
    
    # Load the package. If it doesn't exists, install and load.
    if(!require(lib, character.only = TRUE)) {
      
      # Install the package
      install.packages(lib)
      
      # Load the package
      library(lib, character.only = TRUE)
    }
  })}

# Load the libraries used in this section
libs <- c("tidyverse", "icesTAF", "readr", 
          "lubridate", "caret")

load_lib(libs)

#---------------------------------------
# Download and import the datasets
#---------------------------------------

# Download the datasets from UCI repository
if(!dir.exists("data")) mkdir("data")
if(!file.exists("data/winequality-red.csv")) 
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", "data/winequality-red.csv")
if(!file.exists("data/winequality-white.csv")) 
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", "data/winequality-white.csv")
if(!file.exists("data/winequality.names")) 
  download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names", "data/winequality.names")

# Import the datasets.
# 'red' is the red wine dataset
# 'white' is the white wine dataset.
red   <- read_delim("data/winequality-red.csv", 
                    delim = ";", 
                    locale = locale(decimal_mark = ".", 
                                    grouping_mark = ","), 
                    col_names = TRUE)
white <- read_delim("data/winequality-white.csv", 
                    delim = ";", 
                    locale = locale(decimal_mark = ".", 
                                    grouping_mark = ","), 
                    col_names = TRUE)

# Set column names
cnames <- c("fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH",
            "sulphates", "alcohol", "quality")

# Columns used for prediction are all columns
# except 'quality'.
xcol <- c("fixed_acidity", "volatile_acidity", "citric_acid",
          "residual_sugar", "chlorides", "free_sulfur_dioxide",
          "total_sulfur_dioxide", "density", "pH",
          "sulphates", "alcohol")

colnames(red)   <- cnames
colnames(white) <- cnames

# Add the column 'type' to define the type of wine
red   <- mutate(red,   type = "red")
white <- mutate(white, type = "white")

# Join 'red' and 'white' datasets
wine <- rbind(red, white)
wine <- mutate(wine, 
               quality = as.factor(quality),
               type = as.factor(type))
levels(wine$quality) <- paste0("Q", levels(wine$quality))

#---------------------------------------
# Create train and test sets
#---------------------------------------
# Test set will be 10% of the entire dataset
set.seed(2020, sample.kind = "Rounding")
test_index <- createDataPartition(y = wine$type, 
                                  times = 1, 
                                  p = 0.1, 
                                  list = FALSE)

# Train and test sets for wine type
train_set <- wine[-test_index,]
test_set  <- wine[test_index,]

# Train and test sets for red wine quality
train_set_red <- train_set[which(train_set$type == "red"),]
test_set_red  <- test_set[which(test_set$type == "red"),]

train_set_red$quality <- factor(train_set_red$quality)
test_set_red$quality  <- factor(test_set_red$quality)

#========================================
# Data Explorations
#========================================
# After importing the dataset, it's good practice to check the data.
# Here, we make some basic data checking.


# Check for empty values (NAs) in the dataset
sum(is.na(wine))


# Identification of near zero variance predictors
nearZeroVar(train_set[, xcol], saveMetrics = TRUE)

# Compactly Display the Structure of an Arbitrary R Object
str(train_set)

# Statistics summary
summary(train_set)

#---------------------------------------
# Distribution of outcomes
#---------------------------------------
# We make plots of the distribution of outcomes to 
# understand the data.

# Distribution of red and white wines
ggplot(data = train_set) + 
  geom_bar(aes(type, fill = type)) +
  labs(title = "Prevalence of red and white wines",
       caption = "Source: train_set dataset.") +
  theme(legend.position = 'none')

#---------------------------------------
# Download stats of annual production of wines
# to compare the prevalence of red/white wines with the dataset.
#---------------------------------------

# Load libraries in a separate chunk to hide code and avoid messages.
# If this is joined with the next chunk, the table is hidden
load_lib(c("readxl", "huxtable", "viridis", "ggthemes"))

# The 'huxtable' package creates beautiful tables.
# 'viridis' and 'ggthemes' have color paletes for color blind people

# Download stats file from vinho verde official portal
if(!file.exists("data/vv-stats.xls")) 
  download.file("https://portal.vinhoverde.pt/pt/file/c/1614",
                "data/vv-stats.xls",
                cacheOK = FALSE,
                method = "auto",
                mode = "wb")

# Import stats file
vv_stats <- read_excel(path = "data/vv-stats.xls",
                       sheet = "vinho",
                       range = "A6:D16")

# Calculate the prevalence of red wine
vv_stats <- vv_stats[2:nrow(vv_stats),c(1,3:4)] %>% 
  mutate(Prevalence = 100 * TINTO / BRANCO)

# Create a table with the values.
# Change column names
colnames(vv_stats) <- c("Year", "White", "Red", "Red Prevalence (%)")
vv_stats <- as_hux(vv_stats)

vv_stats <- huxtable::add_colnames(vv_stats)

vv_stats <- vv_stats %>%
  # Format header row
  set_bold(row = 1, col = 1:ncol(vv_stats), value = TRUE)        %>%
  set_top_border(row = 1, col = 1:ncol(vv_stats), value = 1)     %>%
  set_bottom_border(row = c(1,10), col = 1:ncol(vv_stats), value = 1)  %>%
  # Format cells
  set_align(row = 1:4, col = 2, value = 'right')                 %>%
  set_number_format(row = 1:nrow(vv_stats), col = c(2,3), 
                    value = list(function(x)
                      prettyNum(x, big.mark = ",",
                                scientific = FALSE)))            %>% 
  set_number_format(row = 1:nrow(vv_stats), col = 4, value = 2)  %>% 
  # Format table
  set_width(value = 0.6) %>%
  set_caption("Vinho Verde Annual Production 1999-2008")         %>%
  set_position(value = "center")

# Show the table
vv_stats

# Create a plot with the downloaded data.
# The plot is easier to see the values than in the table
# Distribution of quality values
ggplot(data = train_set_red, 
       aes(x = quality, fill ='red')) +
  geom_bar() +
  theme(legend.position="none") +
  labs(title = "Distribution of quality for red wine",
       caption = "Source: train_set_red dataset")

#---------------------------------------
# Variable importance
#---------------------------------------
# The variable importance gives an estimate of the predictive power
# of each feature. 
# Check the help file for 'filterVarImp' for more information.

# Variable importance for wine type
hux(Feature = rownames(filterVarImp(x = train_set[,xcol], 
                                    y = train_set$type)),
    Red   = filterVarImp(x = train_set[,xcol], 
                         y = train_set$type)$red,
    White = filterVarImp(x = train_set[,xcol],
                         y = train_set$type)$white,
    add_colnames = TRUE) %>%
  arrange(desc(Red)) %>% 
  set_bold(row = 1, everywhere, value = TRUE)          %>%
  set_top_border(row = 1, everywhere, value = 1)    %>%
  set_bottom_border(row = c(1,12), everywhere, value = 1)    %>%
  set_align(row = everywhere, col = 2:3, value = 'right') %>%
  set_caption('Variable Importance for Wine Type') %>%
  set_position(value = "center")


#------------------
# Variable importance for red wine quality
#------------------
x <- train_set_red[,xcol]
y <- train_set_red$quality

hux(Feature = rownames(filterVarImp(x = x, y = y)),
    filterVarImp(x = x, y = y),
    add_colnames = TRUE) %>%
  # Format header row
  set_bold(row = 1, everywhere, value = TRUE)          %>%
  set_top_border(row = 1, everywhere, value = 1)       %>%
  set_bottom_border(row = c(1,12), everywhere, value = 1)    %>%
  # Format numbers
  set_number_format(row = 2:12, col = 2:7, value = 3)  %>%
  
  
  map_text_color(row = everywhere, col = 2:7, 
                 by_ranges(seq(0.6, 0.9, 0.1), colorblind_pal()(5))) %>%
  # Format alignment
  set_align(row = everywhere, col = 1,   value = 'left')  %>%
  set_align(row = everywhere, col = 2:7, value = 'right') %>%
  # Title
  set_caption('Variable importance for red Wine quality') %>%
  set_position(value = "center")


# Here we create a plot of variable information of wine quality.
# The same info as in the table above.
# Variable importance for red wine quality
x <- train_set_red[,xcol]
y <- train_set_red$quality
y <- factor(y)

data.frame(Feature = rownames(filterVarImp(x = x, y = y)),
           filterVarImp(x = x, y = y)) %>%
  pivot_longer(col = 2:7, names_to = "Quality",
               values_to = "Value", values_drop_na = TRUE) %>%
  ggplot(aes(x = Feature, y = Value)) +
  geom_col(fill = "red") +
  coord_flip() +
  ggtitle("Variable importance for red wine quality") +
  theme(legend.position = "none") +
  ylab("Relative Importance") +
  geom_hline(yintercept = seq(0.5, 0.9, 0.1), color = "darkgrey") +
  facet_wrap("Quality")


#========================================
# Data visualization
#========================================
# In this section we create several stats plots
# to check the distribution of variables.

# Install and load the libraries used for visualization
# The 'load_lib' function was defined earlier.
load_lib(c("gridExtra", "ggridges", "ggplot2",
           "gtable", "grid", "egg"))


# The 'grid_arrange_shared_legend' function creates a grid of 
# plots with one legend for all plots.
# There's no commentaries because I use the code from the source below.
# Reference: Baptiste Auguié - 2019
# https://cran.r-project.org/web/packages/egg/vignettes/Ecosystem.html
grid_arrange_shared_legend <-
  function(...,
           ncol = length(list(...)),
           nrow = 1,
           position = c("bottom", "right")) {
    
    plots <- list(...)
    position <- match.arg(position)
    g <-
      ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
    legend <- g[[which(sapply(g, function(x)
      x$name) == "guide-box")]]
    lheight <- sum(legend$height)
    lwidth <- sum(legend$width)
    gl <- lapply(plots, function(x)
      x + theme(legend.position = "none"))
    gl <- c(gl, ncol = ncol, nrow = nrow)
    
    combined <- switch(
      position,
      "bottom" = arrangeGrob(
        do.call(arrangeGrob, gl),
        legend,
        ncol = 1,
        heights = unit.c(unit(1, "npc") - lheight, lheight)
      ),
      "right" = arrangeGrob(
        do.call(arrangeGrob, gl),
        legend,
        ncol = 2,
        widths = unit.c(unit(1, "npc") - lwidth, lwidth)
      )
    )
    
    grid.newpage()
    grid.draw(combined)
    
    # return gtable invisibly
    invisible(combined)
    
  }


#------------------
# Density grid
#------------------
# Prediction of red wine type (red or white)
# Create a grid of density plots for each predictor.
# The goal here is to identify features with few distribution overlaps.
dens_grid <- lapply(xcol, FUN=function(var) {
  # Build the plots
  ggplot(train_set) + 
    geom_density(aes_string(x = var, fill = "type"), alpha = 0.5) +
    ggtitle(var)
})
do.call(grid_arrange_shared_legend, args=c(dens_grid, nrow = 4, ncol = 3))


# Another density plots for selected variables.
# These features have low overlaping areas.
# I create this grid for better visualization, since the previous plot has many
# plots and is hard to see.
dens_grid2 <- lapply(c("volatile_acidity", "chlorides", "total_sulfur_dioxide"), 
                     FUN=function(var) {
                       
                       # Build the plots
                       ggplot(train_set) + 
                         geom_density(aes_string(x = var, fill = "type"), alpha = 0.5) +
                         ggtitle(var)
                     })
do.call(grid_arrange_shared_legend, args=c(dens_grid2, nrow = 2, ncol = 2))


# Density plots for another 3 variables.
# These features have large overapping areas.
dens_grid3 <- lapply(c("alcohol", "pH", "citric_acid"), FUN=function(var) {
  
  # Build the plots
  ggplot(train_set) + 
    geom_density(aes_string(x = var, fill = "type"), alpha = 0.5) +
    ggtitle(var)
})
do.call(grid_arrange_shared_legend, args=c(dens_grid3, nrow = 2, ncol = 2))

#------------------
# Box plots
#------------------
# Prediction of wine quality.
# Now we try to find the relationship between 'quality' and 
# the features.

# Arrange the dataset
train_set_red[,cnames] %>% pivot_longer(cols = -12, 
                                        names_to = "Feature", 
                                        values_to = "Value") %>%
  # Create the box plot
  ggplot(aes(x = quality, y= Value, fill = quality)) +
  geom_boxplot() +
  # Format labels
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("") +
  ggtitle("Red wine quality by feature") +
  # Create grid by feature
  facet_wrap(. ~ Feature, scales = "free")

# The distribution of predictors overlap for all quality levels.
# Maybe if we group the quality levels there's less overlap.
# Here we create 2 datasets to predict wine quality on the new levels.
train_set_red <- train_set_red %>% 
  mutate(quality2 = factor(case_when(
    quality %in% c("Q3", "Q4") ~ "low",
    quality %in% c("Q5", "Q6") ~ "medium",
    quality %in% c("Q7", "Q8") ~ "high"),
    levels = c("low", "medium", "high")))

test_set_red <- test_set_red %>% 
  mutate(quality2 = factor(case_when(
    quality %in% c("Q3", "Q4") ~ "low",
    quality %in% c("Q5", "Q6") ~ "medium",
    quality %in% c("Q7", "Q8") ~ "high"),
    levels = c("low", "medium", "high")))

# Plot the distribution of new quality levels
train_set_red %>% ggplot(aes(quality2, fill = quality2)) + geom_bar()



# Now we try to find the relationship between 'quality' and 
# the features.
#
# Another boxplot to check if the grouping improved the overlaps.
# Arrange the dataset
train_set_red[,c(cnames, "quality2")] %>% 
  pivot_longer(cols = -c(12:13), 
               names_to = "Feature", 
               values_to = "Value") %>%
  # Create the box plot
  ggplot(aes(x = quality2, y= Value, fill = quality2)) +
  geom_boxplot() +
  # Format labels
  #  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("") +
  ggtitle("Red wine quality by feature") +
  # Create grid by feature
  facet_wrap(. ~ Feature, scales = "free", ncol = 3, shrink = FALSE)


#------------------
# Density ridge plots
#------------------
# Predict red wine quality
# It's easier to see the distribution with density ridge plots.
# It plots each quality level in a different row.
lapply(xcol, FUN=function(var) {
  
  train_set_red %>% 
    ggplot(data = ., aes_string(x = var, 
                                y = "quality", 
                                fill = "quality", 
                                alpha = 0.5)) + 
    geom_density_ridges() +
    theme_ridges() +
    theme(axis.text.x = element_text(hjust = 1)) +
    scale_fill_brewer(palette = 4) +
    ggtitle(paste0("Red wine quality by ", var))
})

#------------------
# QQ plots
#------------------
# QQ Plots help identify if the feature is normally distributed.
#
# Create a grid for each wine type
# 'rw' = red / white
qq_grid <- lapply(xcol, FUN=function(var) {
  train_set_red %>% 
    dplyr::select(var) %>%
    ggplot(data = ., aes(sample = scale(.))) + 
    stat_qq() +
    stat_qq_line(colour = "red") +
    theme(axis.text.x = element_text(hjust = 1)) +
    ggtitle(var)
})
do.call(grid.arrange, args=c(qq_grid, list(ncol=3)))

#------------------
# Correleation
#------------------
# We want features with low correlation with each other.
# Load the "corrgram" package to draw a correlogram
load_lib("corrgram")

# Draw a correlogram
corrgram(train_set[,xcol], order=TRUE, 
         lower.panel = panel.shade, 
         upper.panel = panel.cor, 
         text.panel  = panel.txt,
         main = "Correlogram: Wine Physicochemical Properties",
         col.regions=colorRampPalette(c("darkgoldenrod4", "burlywood1",
                                        "darkkhaki", "darkgreen")))

# The correlogram has many information, so we filter only
# the features with high correlation and show in a table.
# 'High' correlation here is above 0.5 and lower than -0.5.
#
# Load the 'huxtable' package to format tables
load_lib("huxtable")
options(huxtable.knit_print_df = FALSE)

# Column names, same as 'xcol' but more beautiful
var_names <- c("Fixed acidity", "Volatile acidity", "Citric acid",
               "Residual sugar", "Chlorides", "Free sulfur dioxide",
               "Total sulfur dioxide", "Density", "pH", "Sulphates",
               "Alcohol")

# Calculate the correlation of all predictors
my_cor <- as.data.frame(cor(train_set[,xcol]))

# Row (r) and column (c) numbers of the correlation matrix
# filtered by high correlated features (cor <= -0.5 or cor >= 0.5)
r <- which(my_cor <=-0.5 | my_cor >= 0.5 & my_cor != 1, arr.ind=TRUE)[,"row"]
c <- which(my_cor <=-0.5 | my_cor >= 0.5 & my_cor != 1, arr.ind=TRUE)[,"col"]

# Create a table with high correlations features only
my_cor_hux <- hux(`Feature 1` = var_names[r], 
                  `Feature 2` = var_names[c], 
                  Correlation = sapply(1:length(r), function(x)
                    my_cor[r[x],c[x]]),
                  add_colnames = TRUE) %>%
  # Format the table
  set_bold(row = 1, everywhere, value = TRUE)          %>%
  set_top_border(row = 1, everywhere, value = 1)    %>%
  set_bottom_border(row = c(1,7), everywhere, value = 1)    %>%
  set_align(row = 1, col = 2:3, value = 'center') %>%
  set_number_format(row = 2:7, col = 3, value = 3)              %>% 
  set_caption('High Correlated Features') %>%
  set_position(value = "center")

# Show the table
my_cor_hux


# Previously, we identified 3 features that may be used in prediction
# of wine type. We create a table to check if the correlation 
# between each pair is low.


# Calculate the correlations for volatile acid, chlorides and total sulfur dioxide

# Variable names
xpred1 <- c("volatile_acidity", "chlorides", "total_sulfur_dioxide")

# Nice variable names
xpred2 <- c("Volatile acidity", "Chlorides", "Total sulfur dioxide")

# Calculate the correlation of all predictors
my_cor2 <- as.data.frame(cor(train_set[,xpred1]))

# Create a table with high correlations features only
my_cor_hux2 <- hux(cor(train_set[,xpred1]),
                   add_colnames  = FALSE, 
                   add_rownames = FALSE) 

# Set row and column names
rownames(my_cor_hux2) <- xpred2
colnames(my_cor_hux2) <- xpred2
my_cor_hux2 <- add_rownames(my_cor_hux2, colname = "Feature")
my_cor_hux2 <- add_colnames(my_cor_hux2, value = TRUE)

# Format the table
my_cor_hux2 <- my_cor_hux2 %>%
  set_bold(row = 1, everywhere, value = TRUE)          %>%
  set_top_border(row = 1, everywhere, value = 1)    %>%
  set_bottom_border(row = c(1,4), everywhere, value = 1)    %>%
  set_align(row = 1, col = 2:ncol(my_cor_hux), value = 'center') %>%
  set_number_format(everywhere, everywhere, value = 3)              %>% 
  set_caption('Correlation Matrix - Selected Features') %>%
  set_position(value = "center") %>%
  set_width(value = 0.6)

# Show the table
my_cor_hux2


#========================================
# Modeling and results
#========================================
# Here we make predictions with information gained from
# data exploration and visualization.
#

# Formula used in predictions
fml <- as.formula(paste("type", "~", 
                        paste(xcol, collapse=' + ')))


#------------------
# Single predictor
#------------------
# Predict wine type with total_sulfur_dioxide + chlorides + volatile_acidity
# The first prediction is very simple. We predict 'red' if
# the feature value is above a certain cutoff value, and 'white' 
# otherwise. 
# We do this for the 3 best features discorevered in data exploration.
# Then we combine the results in a single ensemble.
#
# Create a list with variable names and cutoff decision rule.
# If the predicted value is lower than the cutoff value, the first color
# is chosen, otherwise the second. To understand this, look at the
# density plots in data visualization.
type_var <- list( c("white", "red"), c("white", "red"), c("red", "white"))
names(type_var) <- c("volatile_acidity", "chlorides", "total_sulfur_dioxide")

# Create an empty results table. The first row
# contains NAs and will be removed after the predictions.
type_results <<- data.frame(Feature = NA,
                            Accuracy = NA,
                            Sensitivity = NA,
                            Specificity = NA,
                            stringsAsFactors = FALSE)

# Prediction function
preds <- sapply(1:length(type_var), function(x){
  
  # Get the variable name
  var <- names(type_var[x])
  
  # Cutoff value is the distribution range divided by 500
  cutoff <- seq(min(train_set[,var]), 
                max(train_set[,var]), 
                length.out = 500)
  
  # Calculate accuracy
  acc <- map_dbl(cutoff, function(y){
    type <- ifelse(train_set[,var] < y, type_var[[x]][1], 
                   type_var[[x]][2]) %>% 
      factor(levels = levels(train_set$type))
    
    # Accuracy
    mean(type == train_set$type)
  })
  
  # Build the accuracy vs cutoff curve
  acc_plot <- data.frame(cutoff = cutoff, Accuracy = acc) %>%
    ggplot(aes(x = cutoff, y = Accuracy)) + 
    geom_point() +
    ggtitle(paste0("Accuracy curve for ", var))
  
  # Print the plot
  print(acc_plot)
  
  # Predict new values in the test set
  # The model uses the cutoff value with the best accuracy.
  max_cutoff <- cutoff[which.max(acc)]
  y_hat <- ifelse(test_set[,var] < max_cutoff,
                  type_var[[x]][1], type_var[[x]][2]) %>% 
    factor(levels = levels(test_set$type))
  
  # Calculate accuracy, specificity and sensitivity
  acc <- max(acc)
  sens <- sensitivity(y_hat, test_set$type)
  spec <- specificity(y_hat, test_set$type)
  
  # Update results table
  type_results <<- rbind(type_results,
                         data.frame(Feature = names(type_var[x]),
                                    Accuracy = acc,
                                    Sensitivity = sens,
                                    Specificity = spec,
                                    stringsAsFactors = FALSE))
  
  # The prediction will be used in the ensemble
  return(y_hat)
})  

# Remove first row with NA
type_results <- type_results[2:nrow(type_results),]

# Combine the results using majority of votes
y_hat_ens <-as_factor(data.frame(preds) %>%
                        mutate(x = as.numeric(preds[,1] == "red") + 
                                 as.numeric(preds[,2] == "red") + 
                                 as.numeric(preds[,3]  == "red"),
                               y_hat = ifelse(x >=2, "red", "white")) %>%
                        pull(y_hat))

# Update results table
type_results <<- rbind(type_results,
                       data.frame(Feature = "Ensemble",
                                  Accuracy = mean(y_hat_ens == test_set$type),
                                  Sensitivity = sensitivity(y_hat_ens, test_set$type),
                                  Specificity = specificity(y_hat_ens, test_set$type),
                                  stringsAsFactors = FALSE))
# Show the results table
as_hux(type_results, 
       add_colnames = TRUE) %>%
  # Format header
  set_bold(row = 1, col = everywhere, value = TRUE)        %>%
  set_top_border(row = 1, col = everywhere, value = 1)     %>%
  set_bottom_border(row = c(1,5), col = everywhere, value = 1)  %>%
  # Format cells
  set_align(row = 1:4, col = 2, value = 'right')                %>%
  # Format numbers
  set_number_format(row = everywhere, col = 2:4, value = 3)     %>% 
  # Format table
  set_caption("Superior Performance for Combined Predictions")  %>%
  set_position(value = "center")



#------------------
# Linear Regression
#------------------
# Predict wine type with total_sulfur_dioxide + chlorides + volatile_acidity

# Train the linear regression model
fit_lm <- train_set %>% 
  # Convert the outcome to numeric
  mutate(type = ifelse(type == "red", 1, 0)) %>%
  # Fit the model
  lm(type ~ total_sulfur_dioxide + chlorides + volatile_acidity, data = .)

# Predict
p_hat_lm <- predict(fit_lm, newdata = test_set)

# Convert the predicted value to factor
y_hat_lm <- factor(ifelse(p_hat_lm > 0.5, "red", "white"))

# Evaluate the results
caret::confusionMatrix(y_hat_lm, test_set$type)


#------------------
# Knn
#------------------
# Predict wine type with all features
# Train
fit_knn <- knn3(formula = fml, data = train_set, k = 5)

# Predict
y_knn <- predict(object = fit_knn, 
                 newdata = test_set, 
                 type ="class")

# Compare the results: confusion matrix
caret::confusionMatrix(data = y_knn, 
                       reference = test_set$type, 
                       positive = "red")

# F1 score
F_meas(data = y_knn, reference = test_set$type)


#------------------
# Regression tree
#------------------
# Predict wine type with all features

# The "rpart" package trains regression trees and 
# "rpart.plot" plots the tree
load_lib(c("rpart", "rpart.plot"))

# Train the model
fit_rpart <- rpart::rpart(formula = fml, 
                          method = "class", 
                          data = train_set)
# Predict
y_rpart <- predict(object = fit_rpart, 
                   newdata = test_set, 
                   type = "class")

# Compare the results: confusion matrix
caret::confusionMatrix(data = y_rpart, 
                       reference = test_set$type, 
                       positive = "red")

# Plot the result
rpart.plot(fit_rpart)

# F1 score
F_meas(data = y_rpart, reference = test_set$type)

# Variable importance
caret::varImp(fit_rpart)


#------------------
# Random Forest
#------------------
# Predict wine type with all features

# The "randomForest" package trains classification and regression
# with Random Forest
load_lib("randomForest")

# Train the model
fit_rf <- randomForest(formula = fml, data = train_set)

# Predict
y_rf <- predict(object = fit_rf, newdata = test_set)

# Compare the results: confusion matrix
caret::confusionMatrix(data = y_rf, 
                       reference = test_set$type, 
                       positive = "red")

# F1 score
F_meas(data = y_rf, reference = test_set$type)

# Plot the error curve
data.frame(fit_rf$err.rate) %>% mutate(x = 1:500 ) %>% 
  ggplot(aes(x = x)) + 
  #  geom_line(aes(y = OOB)) +
  geom_line(aes(y = red),   col = "red") +
  geom_line(aes(y = white), col = "blue") +
  ggtitle("Random Forest Error Curve") +
  ylab("Error") +
  xlab("Number of trees") +
  geom_text(aes(x = 70,  y = 0.02), label = "Red wine", col = "red") + 
  #  geom_text(aes(x = 100, y = 0.01), label = "Error") +
  geom_text(aes(x = 100, y = 0), label = "White wine", col = "blue")

# Variable importance plot
varImpPlot(fit_rf, main = "Random Forest Variable importance")


#------------------
# LDA
#------------------
# Predict wine type with all features
load_lib("MASS")

# Train the model
fit_lda <- lda(formula = fml, data = train_set)

# Predict
y_lda <- predict(object = fit_lda, newdata = test_set)

# Compare the results: confusion matrix
caret::confusionMatrix(data = y_lda[[1]], 
                       reference = test_set$type, 
                       positive = "red")



# F1 score
F_meas(data = y_lda[[1]], reference = test_set$type)

# Plot the result
plot(fit_lda)

#------------------
# QDA
#------------------
# Predict wine type with all features
load_lib(c("MASS", "scales"))

# Train the model
fit_qda <- qda(formula = fml, data = train_set)

# Predict
y_qda <- predict(object = fit_qda, newdata = test_set)

# Compare the results: confusion matrix
caret::confusionMatrix(data = y_qda[[1]], 
                       reference = test_set$type, 
                       positive = "red")
# F1 score
F_meas(data = y_qda[[1]], reference = test_set$type)


data.frame(Model = c("Single predictor", "Linear Regression", "Knn", 
                     "Regression trees", "Random forest",
                     "LDA", "QDA"),
           Accuracy = c(percent(mean(y_hat_ens  == test_set$type), accuracy = 0.1),
                        percent(mean(y_hat_lm   == test_set$type), accuracy = 0.1),
                        percent(mean(y_knn      == test_set$type), accuracy = 0.1),
                        percent(mean(y_rpart    == test_set$type), accuracy = 0.1),
                        percent(mean(y_rf       == test_set$type), accuracy = 0.1),
                        percent(mean(y_lda[[1]] == test_set$type), accuracy = 0.1),
                        percent(mean(y_qda[[1]] == test_set$type), accuracy = 0.1)),
           
           Sensitivity = c(percent(sensitivity(y_hat_ens,  test_set$type), accuracy = 0.1),
                           percent(sensitivity(y_hat_lm,   test_set$type), accuracy = 0.1),
                           percent(sensitivity(y_knn,      test_set$type), accuracy = 0.1), 
                           percent(sensitivity(y_rpart,    test_set$type), accuracy = 0.1), 
                           percent(sensitivity(y_rf,       test_set$type), accuracy = 0.1), 
                           percent(sensitivity(y_lda[[1]], test_set$type), accuracy = 0.1), 
                           percent(sensitivity(y_qda[[1]], test_set$type), accuracy = 0.1)), 
           
           Specificity = c(percent(specificity(y_hat_ens,  test_set$type), accuracy = 0.1),
                           percent(specificity(y_hat_lm,   test_set$type), accuracy = 0.1),
                           percent(specificity(y_knn,      test_set$type), accuracy = 0.1),
                           percent(specificity(y_rpart,    test_set$type), accuracy = 0.1),
                           percent(specificity(y_rf,       test_set$type), accuracy = 0.1),
                           percent(specificity(y_lda[[1]], test_set$type), accuracy = 0.1),
                           percent(specificity(y_qda[[1]], test_set$type), accuracy = 0.1)))

#------------------
# Cross validation (train) and ensemble
#------------------
# Now we are going to do several things:
# 1. train 10 classification models,
# 2. make the predictions for each model
# 3. calculate some statistics and store in the 'results' table
# 4. plot the ROC and precision-recall curves
# Then, we're going to plot the values in the 'results' table
# and make the ensemble of all models together.

# Load the packages used in this section
# Package "pROC" creates ROC and precision-recall plots
load_lib(c("pROC", "plotROC"))

# Several machine learning libraries
load_lib(c("e1071", "dplyr", "fastAdaboost", "gam", 
           "gbm", "import", "kernlab", "kknn", "klaR", 
           "MASS", "mboost", "mgcv", "monmlp", "naivebayes", "nnet", "plyr", 
           "ranger", "randomForest", "Rborist", "RSNNS", "wsrf"))


# Define models
models <- c("glm", "lda", "naive_bayes", "svmLinear", "rpart",
            "knn", "gamLoess", "multinom", "qda", "rf", "adaboost")

# We run cross validation in 10 folds, training with 90% of the data.
# We save the prediction to calculate the ROC and precision-recall curves
# and we use twoClassSummary to compute the sensitivity, specificity and 
# area under the ROC curve
control <- trainControl(method = "cv", number = 10, p = .9,
                        summaryFunction = twoClassSummary, 
                        classProbs = TRUE,
                        savePredictions = TRUE)

control <- trainControl(method = "cv", number = 10, p = .9,
                        classProbs = TRUE,
                        savePredictions = TRUE)

# Create 'results' table. The first row
# contains NAs and will be removed after
# the training
results <- tibble(Model = NA,
                  Accuracy = NA,
                  Sensitivity = NA,
                  Specificity = NA,
                  F1_Score = NA,
                  AUC = NA)
#-------------------------------
# Start parallel processing
#-------------------------------
# The 'train' function in the 'caret' package allows the use of
# parallel processing. Here we enable this before training the models.
# See this link for details:
# http://topepo.github.io/caret/parallel-processing.html
cores <- 4    # Number of CPU cores to use
# Load 'doParallel' package for parallel processing
load_lib("doParallel")
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)


set.seed(1234, sample.kind = "Rounding")
# Formula used in predictions
fml <- as.formula(paste("type", "~", 
                        paste(xcol, collapse=' + ')))

# Run predictions
preds <- sapply(models, function(model){ 
  
  if (model == "knn") {
    # knn use custom tuning parameters
    grid <- data.frame(k = seq(3, 50, 2))
    fit <- caret::train(form = fml, 
                        method = model, 
                        data = train_set, 
                        trControl = control,
                        tuneGrid = grid)
  } else if (model == "rf") {
    # Random forest use custom tuning parameters
    grid <- data.frame(mtry = c(1, 2, 3, 4, 5, 10, 25, 50, 100))
    
    fit <- caret::train(form = fml,
                        method = "rf", 
                        data = train_set,
                        trControl = control,
                        ntree = 150,
                        tuneGrid = grid,
                        nSamp = 5000)
  } else {
    # Other models use standard parameters (no tuning)
    fit <- caret::train(form = fml, 
                        method = model, 
                        data = train_set, 
                        trControl = control)
  }
  
  # Predictions
  pred <- predict(object = fit, newdata = test_set)
  
  # Accuracy
  acc <- mean(pred == test_set$type)
  
  # Sensitivity
  sen <- sensitivity(data = pred, 
                     reference = test_set$type, 
                     positive = "red")
  # Specificity
  spe <- specificity(data = pred, 
                     reference = test_set$type, 
                     positive = "red")
  
  # F1 score
  f1 <- F_meas(data = factor(pred), reference = test_set$type)
  
  # AUC
  auc_val <- auc(fit$pred$obs, fit$pred$red)
  
  # Store stats in 'results' table
  results <<- rbind(results,
                    tibble(
                      Model = model,
                      Accuracy = acc,
                      Sensitivity = sen,
                      Specificity = spe,
                      AUC = auc_val,
                      F1_Score = f1))
  
  # The predictions will be used for ensemble
  return(pred)
}) 

# Remove the first row of 'results' that contains NAs
results <- results[2:(nrow(results)),]


# Use votes method to ensemble the predictions
votes <- rowMeans(preds == "red")
y_hat <- factor(ifelse(votes > 0.5, "red", "white"))

# Update the 'results' table
results <<- rbind(results,
                  tibble(
                    Model = "Ensemble",
                    Accuracy = mean(y_hat == test_set$type),
                    Sensitivity = sensitivity(y_hat, test_set$type),
                    Specificity = specificity(y_hat, test_set$type),
                    AUC = auc(y_hat, as.numeric(test_set$type)),
                    F1_Score = F_meas(y_hat, test_set$type)))

# Show the results table
as_hux(results,
       add_colnames = TRUE) %>%
  # Format header row
  set_bold(row = 1, everywhere, value = TRUE)          %>%
  set_top_border(row = 1, everywhere, value = 1)       %>%
  set_bottom_border(row = c(1,13), everywhere, value = 1)    %>%
  # Format numbers
  set_number_format(row = -1, col = 2:6, value = 3)  %>%
  # Format alignment
  set_align(row = everywhere, col = 1,   value = 'left')  %>%
  set_align(row = everywhere, col = 2:6, value = 'right') %>%
  # Title
  set_caption('Model Performance With Cross Validation') %>%
  set_position(value = "center")

# Generalized linear model (glm) has the best overall performance,
# better even than the ensemble.

hux(Accuracy  = results[which.max(results$Accuracy),1]$Model,
    Sensitivity = results[which.max(results$Sensitivity),1]$Model,
    Specificity = results[which.max(results$Specificity),1]$Model,
    F_1   = results[which.max(results$F1_Score),1]$Model,
    AUC  = results[which.max(results$AUC),1]$Model,
    add_colnames = TRUE) %>%
  # Format header row
  set_bold(row = 1, col = everywhere, value = TRUE)        %>%
  set_top_border(row = 1, col = everywhere, value = 1)     %>%
  set_bottom_border(row = c(1,2), col = everywhere, value = 1)  %>%
  # Format table
  set_width(value = 0.6)                                   %>%
  set_caption("Best model")                                %>%
  set_position(value = "center")


#-------------------------------
# Plot the 'results' table
#-------------------------------
# We create a grid with all plots together.
# Each plot is simple Model vs Stats
results %>% 
  # Convert columns to lines
  pivot_longer(cols = 2:6, names_to = "Metric", values_drop_na = TRUE) %>%
  ggplot(aes(x = Model, y = value, group = 1)) + 
  geom_line() +
  geom_point() +
  # Y axis scale
  ylim(0.75, 1) +
  # Format labels
  ggtitle("Model performance") + 
  ylab("") +
  theme(legend.position="none" ,
        axis.text.x = element_text(angle = 90)) +
  # Arrange in grid
  facet_wrap(~Metric)



# ================================
# Predict red wine quality
# ================================


set.seed(1234, sample.kind = "Rounding")
# Formula used in predictions
fml_qual <- as.formula(paste("quality2", "~", 
                             paste(xcol, collapse=' + ')))

# Define models
#"glm",gamLoess, qda, adaboost
models <- c( "lda", "naive_bayes", "svmLinear", "rpart",
             "knn", "multinom", "rf")

#train_set_red <- train_set[which(train_set$type == "red"),]
#test_set_red  <- test_set[which(test_set$type == "red"),]
#train_set_red$quality <- factor(train_set_red$quality)
#test_set_red$quality  <- factor(test_set_red$quality)
# Create 'results' table. The first row
# contains NAs and will be removed after
# the training
quality_results <- tibble(Model = NA,
                          Quality = NA,
                          Accuracy = NA,
                          Sensitivity = NA,
                          Specificity = NA,
                          F1_Score = NA)

preds_qual <- sapply(models, function(model){ 
  
  print(model)
  if (model == "knn") {
    # knn use custom tuning parameters
    grid <- data.frame(k = seq(3, 50, 2))
    fit <- caret::train(form = fml_qual, 
                        method = model, 
                        data = train_set_red, 
                        trControl = control,
                        tuneGrid = grid)
  } else if (model == "rf") {
    # Random forest use custom tuning parameters
    grid <- data.frame(mtry = c(1, 2, 3, 4, 5, 10, 25, 50, 100))
    
    fit <- caret::train(form = fml_qual,
                        method = "rf", 
                        data = train_set_red,
                        trControl = control,
                        ntree = 150,
                        tuneGrid = grid,
                        nSamp = 5000)
  } else {
    # Other models use standard parameters (no tuning)
    fit <- caret::train(form = fml_qual, 
                        method = model, 
                        data = train_set_red, 
                        trControl = control)
  }
  
  # Predictions
  pred <- predict(object = fit, newdata = test_set_red)
  
  # Accuracy
  acc <- mean(pred == test_set_red$quality2)
  
  # Sensitivity
  sen <- caret::confusionMatrix(pred,
                                test_set_red$quality2)$byClass[,"Sensitivity"]
  # Specificity
  spe <- caret::confusionMatrix(pred,
                                test_set_red$quality2)$byClass[,"Specificity"]
  
  # F1 score
  f1 <- caret::confusionMatrix(pred,
                               test_set_red$quality2)$byClass[,"F1"]
  
  # Store stats in 'results' table
  quality_results <<- rbind(quality_results,
                            tibble(Model = model, 
                                   Quality = levels(test_set_red$quality2),
                                   Accuracy = acc,
                                   Sensitivity = sen,
                                   Specificity = spe,
                                   F1_Score = f1))
  
  # The predictions will be used for ensemble
  return(pred)
}) 

# Remove the first row of 'results' that contains NAs
quality_results <- quality_results[2:(nrow(quality_results)),]



#-------------------------------
# Combine all models
#-------------------------------
# Use votes method to ensemble the predictions
votes <- data.frame(low    = rowSums(preds_qual =="low"),
                    medium = rowSums(preds_qual =="medium"),
                    high   = rowSums(preds_qual =="high"))

y_hat <- factor(sapply(1:nrow(votes), function(x)
  colnames(votes[which.max(votes[x,])])))

y_hat <- relevel(y_hat, "medium")

# Accuracy
acc <- caret::confusionMatrix(y_hat,
                              test_set_red$quality2)$overall["Accuracy"]

# Sensitivity
sen <- caret::confusionMatrix(y_hat,
                              test_set_red$quality2)$byClass[,"Sensitivity"]
# Specificity
spe <- caret::confusionMatrix(y_hat,
                              test_set_red$quality2)$byClass[,"Specificity"]

# F1 score
f1 <- caret::confusionMatrix(y_hat,
                             test_set_red$quality2)$byClass[,"F1"]


quality_results <<- rbind(quality_results,
                          tibble(Model = "Ensemble",
                                 Quality = levels(test_set_red$quality2),
                                 Accuracy = acc,
                                 Sensitivity = sen,
                                 Specificity = spe,
                                 F1_Score = f1))


#-------------------------------
# Plot the 'results' table
#-------------------------------
# Make a plot for each metric with the 3 quality levels
# The plots are grouped using the chunk options.

# Metric names
metrics <- names(quality_results)[3:6]

res <- sapply(metrics, function(var){
  
  # Plot stored in 'p'
  p <- quality_results %>% 
    # Convert columns to lines
    pivot_longer(cols = 3:6, names_to = "Metric", 
                 values_drop_na = TRUE) %>%
    
    pivot_wider(names_from = Quality) %>%
    filter(Metric == var) %>%
    
    ggplot(aes(x = Model, group = 1)) + 
    # Draw lines
    geom_line(aes(y = low,     col = "low")) +
    geom_line(aes(y = medium,  col = "medium")) +
    geom_line(aes(y = high,    col = "high")) +
    # Draw points
    geom_point(aes(y = low,    col = "low")) +
    geom_point(aes(y = medium, col = "medium")) +
    geom_point(aes(y = high,   col = "high")) +
    # Format labels
    ggtitle(var) + 
    ylab("Model") +
    theme(axis.text.x = element_text(angle = 90))
  
  # Show the plot
  print(p)
})


# Here we compare the distribution of the predicted and 
# original values.
# Plot of the predicted quality distribtuion
data.frame(Quality = factor(names(summary(y_hat)), 
                            levels = names(summary(y_hat))),
           Count= summary(y_hat)) %>%
  ggplot(aes(x = Quality, y = Count, fill = Quality)) +
  geom_col() +
  theme(legend.position = "none") +
  labs(title = "Predicted quality distribution",
       caption = "Source: predictions.")

# Create the plot of the original quality distribtuion
train_set_red %>% 
  ggplot(aes(quality2, fill = quality2)) + 
  geom_bar() +
  labs(title = "Original quality distribution",
       caption = "Source: train_set_red dataset.")



#-------------------------------
# Stop parallel processing used in 'train'
#-------------------------------
stopCluster(cl)


#================================
# Clustering
#================================


#-------------------------------
# k-means
#-------------------------------
# Reference:
# https://www.datanovia.com/en/lessons/k-means-clustering-in-r-algorith-and-practical-examples/

# Load 'factoextra' to visualize clusters
load_lib("factoextra")

# Determine and visualize the optimal number of clusters 
# using total within sum of square (method = "wss")
train_set %>% filter(type == "red") %>% .[,xcol] %>%
  fviz_nbclust(x = ., FUNcluster = kmeans, method = "wss") + 
  geom_vline(xintercept = 4, linetype =2) +
  scale_y_continuous(labels = comma)


# We use 25 random starts for the clusters
k <- train_set %>% filter(type == "red") %>% .[,xcol] %>%
  kmeans(x = ., centers = 4, nstart = 25)

# Calculate cluster means
cm <- as_hux(data.frame(t(k$centers)), add_rownames = TRUE)
colnames(cm) <- c("Feature", paste("Cluster", 1:4))
cm <- add_colnames(cm)  
cm %>%
  # Format header row
  set_bold(row = 1, col = everywhere, value = TRUE)        %>%
  set_top_border(row = 1, col = everywhere, value = 1)     %>%
  set_bottom_border(row = c(1,12), col = everywhere, value = 1)  %>%
  # Format cells
  set_align(row = everywhere, col = 1,   value = 'left')   %>%
  set_align(row = everywhere, col = 2:5, value = 'right')  %>%
  set_number_format(row = 2:nrow(cm), 
                    col = everywhere, value = 3)           %>% 
  # Format table
  set_width(value = 0.7)                                   %>%
  set_position(value = "center")                           %>%
  set_caption("Cluster Center by Feature")                 


# Plot the cluster
train_set %>% filter(type == "red") %>% .[,xcol] %>%
  fviz_cluster(object = k, 
               choose.vars = c("chlorides", "total_sulfur_dioxide"),
               geom   = "point", 
               repel  = TRUE, 
               main   = "Cluster plot with selected features",
               xlab   = "Chlorides",
               ylab   = "Total Sulfur Dioxide")


