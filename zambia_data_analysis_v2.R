# Author: Liam Timms UQ uqltimm1
# Created: 19/08/2021
# Modified: 01/11/2022
# Plotting total amount of land vs. total cost of land in Zambia
# v2: Zambia and Tanzania datasets.
#     Allow different sizes for area cells (v1 has size=1km2)
# Set the working directory, install packages, read data
setwd(r'(Z:\Elephant_project\Code)')

# Install graphing package, mainly for ggplot2
# library("segmented")
library("tidyverse")

set.seed(42)

# Data sorted by cost
file_name <- 'zambia_data_sorted_duplicated.csv'
# file_name <- 'tanzania_data_sort_duplicated.csv'

# Read in the duplicated real estate data sorted by unit price
read_df <- read.csv(file_name)
# Use a smaller df = every 10th row of original data
land_data_unitprice <- read_df[seq(from=0, to=nrow(read_df), by=10), ]

# Compute cumulative number of area cells based on cell size
cell_size <- 706  # km2. Based on area of circle with radius 10km.

y_label_text <- paste("Cumulative cells (", cell_size, "km\U00B2)", sep='')

land_data_unitprice['Cells_cumsum'] <- land_data_unitprice$Area_cumsum/cell_size

# Create df row for Price_cumsum = 0, Area_cumsum = 0
zero_df <- data.frame(X=c(0), Area=c(0), Price=c(0), Area.per.unit.cost=c(0),
                      Area_cumsum=c(0), Price_cumsum=c(0), Cells_cumsum=c(0))

# Append row of zeros to beginning of land_data_unit_price
land_data_unitprice <- rbind(zero_df, land_data_unitprice)

# ------------- FIT FUNCTION TO COST/AREA DATA ------------- 
# # Remove outliers
# land_data = subset(land_data, Area<45& Price<40000) 

# Regression on cumsum data

# Fit logistic model - nonlinear least squares
# model_logistic <- nls(
#         formula = Cells_cumsum ~ L/(1+ exp(-k*(Price_cumsum-x0))),
#         data = land_data_unitprice,
#         start = c(L=100, k=0.05, x0=100)
# )

# Fit x^a, a<1 - nonlinear least squares
model_root <- nls(
        formula = Cells_cumsum ~ a*Price_cumsum^b,
        data = land_data_unitprice,
        start = c(a=1, b = 1/2)
)

# Fitted logistic model parameters
# L <- summary(model_logistic)$coef[1]
# k <- summary(model_logistic)$coef[2]
# x0 <- summary(model_logistic)$coef[3]

# Fitted ax^b model parameters
pow_a <- summary(model_root)$coef[1]
pow_b <- summary(model_root)$coef[2]

bev_a_init <- tail(land_data_unitprice$Cells_cumsum, 1)*1.05 # Largest cumulative land * 1.05

index <- 40   # Index for numerical derivative
# bev_b = bev_a / numerical derivative at mu=0
bev_b_init <- bev_a_init / ((land_data_unitprice$Cells_cumsum[index] - land_data_unitprice$Cells_cumsum[1])/
                   (land_data_unitprice$Price_cumsum[index] - land_data_unitprice$Price_cumsum[1]))

model_bev <- nls(
        formula = Cells_cumsum ~ a*Price_cumsum/(b+Price_cumsum),
        data = land_data_unitprice,
        start = c(a=bev_a_init, b = bev_b_init) # Set initial guesses for a and b
        )

bev_a <- coef(model_bev)[[1]]
bev_b <- coef(model_bev)[[2]]

# Fit a straight line to the data (used for Kuempel parameters)
model_line <- lm(
                formula = Cells_cumsum ~ Price_cumsum, # Specify zero intercept
                data = land_data_unitprice
                )

straight_inter <- coef(model_line)[[1]]
straight_slope <- coef(model_line)[[2]]

# Create equation text for Beverton-Holt and power functions
bev_holt_text <- paste('A(P) = ', signif(bev_a, 2),'P\\(',
                      signif(bev_b, 2), '+P)', sep=''
                      )

power_text <- paste('A(P) = ', signif(pow_a, 2),'P^',
                      signif(pow_b, 2), sep=''
                    )

straight_text <- paste('A(P) = ', signif(straight_slope, 2),'P', ' + ',
                       signif(straight_inter, 2), sep='')

# ------------------------- PLOT CUMULATIVE AREA VS PRICE -------------------------
# LINEAR SCALE
# Cumsum area vs cumsum price, sorted by area/unit price
area_price <- ggplot(data = land_data_unitprice) +
        # geom_function(# plot fitted logistic function
        # mapping = aes(x = Price_cumsum, colour='Logistic'),
        # fun = function(x) L/(1+ exp(-k*(x-x0))),
        # size = 1.5
        # 
        # ) +
        geom_function(# plot fitted x^1/a function
                mapping = aes(x = Price_cumsum, colour='Power'),
                fun = function(x) pow_a*x^pow_b,
                size = 1.5

        ) +
        geom_function(# plot fitted bev-holt function
                mapping = aes(x = Price_cumsum, colour='Bev.-Holt'),
                fun = function(x) bev_a*x/(bev_b+x),
                size = 1.5

        ) +
        geom_function(# plot fitted straight line
          mapping = aes(x = Price_cumsum, colour='Straight'),
          fun = function(x) kuempel_slope*x,
          size = 1.5
          
        ) +
        geom_point(# plot the data
          mapping = aes(
            x = Price_cumsum,
            y = Cells_cumsum,
            colour='Data'
          )
        ) +
        # Add text for r^2 value, residual std. dev, residual DOF.
        geom_label(
                aes(x=1.5e8, y=250),
                label = paste(
                              # 'Logistic: RSD = ', signif(sigma(model_logistic), 3),
                              # ', DOF = ', summary(model_logistic)$df[2],
                              '\nPower: RSD = ', signif(sigma(model_root), 3),
                              ', DOF = ', summary(model_root)$df[2],
                              '\nBev.-Holt: RSD = ', signif(sigma(model_bev), 3),
                              ', DOF = ', summary(model_bev)$df[2],
                              '\n', bev_holt_text,
                              '\n', power_text,
                              '\n', straight_text
                              ),
        ) +
        # Add axis and title labels
        labs(title="Cumulative area vs. cumulative cost\n (sorted by area/unit cost)\n Zambia",
             x ="Cumulative cost ($USD)",
             y = y_label_text
        ) +
        # Add legend
        scale_colour_manual("",
                            breaks = c("Data", 'Logistic', 'Power', 'Bev.-Holt', 'Straight'),
                            values = c('black', '#1b9e77', '#7570b3', '#d95f02', '#e7298a')
                            )

# LOG-LOG SCALE
# log(Cumsum area) vs log(cumsum price), sorted by area/unit price
# loglog_area_price = ggplot(data=land_data_unitprice,
#                            aes(x=log(Price_cumsum), y=Cells_cumsum)) + 
#   # Add markers on plot for all data points
#   geom_point() +
#   # Add axis and title labels
#   labs(title="Cumulative area vs. log(cumulative money)\n (sorted by area per unit cost)",
#        x ="log(Cumulative money) (log($USD)",
#        y = "Cumulative area (km\U00B2)"
#   )

# Try fitting piecewise linear function to Tanzania data
# x <- land_data_unitprice$Price_cumsum
# y <- land_data_unitprice$Area_cumsum

# # Linear model
# lin_fit <- lm(y~x)
# # Piecewise model with 4 breakpoints (psi is vector of estimated breakpoints)
# seg_fit <- segmented(lin_fit, seg.Z = ~x, npsi=4)
# 
# # Intercept
# a <- seg_fit$coefficients[1]
# s <- slope(seg_fit)$x  # slopes
# 
# # Breakpoints
# b <- seg_fit$psi[,2]
# 
# plot(seg_fit, col="blue")
# points(x, y, cex=0.5)
# # lines(x, lin_fit$coefficients[2]*x+lin_fit$coefficients[1], col="red")
# 
# # lines(x, a[1]+s[1]*x, col='green')
# # lines(x, a[1]+s[1]*b[1]+s[2]*(x-b[1]), col='red')
# # lines(x, a[1]+s[1]*b[1]+s[2]*(b[2]-b[1])+s[3]*(x-b[2]), col='pink')
# # lines(x, a[1]+s[1]*b[1]+s[2]*(b[2]-b[1])+s[3]*(b[3]-b[2])+s[4]*(x-b[3]), col='brown')
# 
# # First breakpoint is at ~230611.4. This corresp. to x[1] - x[132]. 
# # Fit linear model with intercept 0 for this first segment, then use segmented
# # to fit to rest of data
# lin_fit_initial <- lm(y[1:132]~x[1:132])

# Print the plots in window
print(area_price)
