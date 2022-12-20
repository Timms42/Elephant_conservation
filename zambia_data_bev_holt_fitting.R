# Plotting cost of land in Zambia vs. amount of land

# Set the working directory, install packages, read data
# setwd(r'(C:\Users\admin\Documents\University\Honours_project\Code)')

# Install graphing package, mainly for ggplot2
library("tidyverse")
# Install package for LaTeX labels
# library('tikzDevice')

# Output to .text file, set plots dimensions
# tikzDevice::tikz(file = "./plot_linlin.tex", width = 5, height = 3)

setwd("H:\\Honours_project\\Code")

# Plain data
file_name1 = 'zambia_data.csv'

# Data sorted by cost
file_name2 = 'zambia_data_sorted.csv'

# Data sorted by cost
file_name3 = 'zambia_data_sorted2.csv'

# Read in the real estate data
land_data = read.csv(file_name1)

# Read in the sorted real estate data
land_data_sort = read.csv(file_name2)

# Read in the sorted real estate data
land_data_unitprice = read.csv(file_name3)


# ------------- FIT FUNCTION TO COST/AREA DATA ------------- 
# # Remove outliers
# land_data = subset(land_data, Area<45& Price<40000) 

# -------------------------- PLOT CUMSUMS --------------------------
# ----------------- OPTIMAL GUESS FOR BEV-HOLT NLS -----------------
# Regression on cumsum data

bev_a = tail(land_data_unitprice$Area_cumsum, 1)*1.05

# Number of data points
iter = length(land_data_unitprice$Area_cumsum) - 1

# Initialise vectors of fitted a, b parameters, std dev of error, no. of nls iterations
a_list = rep(0, length.out=iter)
b_list = rep(0, length.out=iter)
RSD_list = rep(0, length.out=iter)
numiter_list = rep(0, length.out=iter)


# What is the best initial guess for mu_0?
# Try numerical derivative for all data points
for(ii in 2:iter+1){
  num_deriv_ii = (land_data_unitprice$Area_cumsum[ii] - land_data_unitprice$Area_cumsum[1])/
            (land_data_unitprice$Price_cumsum[ii] - land_data_unitprice$Price_cumsum[1])
  
  bev_b = bev_a/num_deriv_ii
  
  model_bev = nls(
    formula = Area_cumsum ~ a*Price_cumsum/(b+Price_cumsum),
    data = land_data_unitprice,
    start = c(a=bev_a, b = bev_b))
  
  fitted_a = coef(model_bev)[1]
  fitted_b = coef(model_bev)[2]
  
  # Create equation text for logistic plot
  bev_holt_text = paste('A(P) ~= ', signif(fitted_a, 2),'P\\(',
                        signif(fitted_b, 2), '+P)', sep=''
  )
  
  a_list[ii-1] = fitted_a    # fitted A_max value
  b_list[ii-1] = fitted_b    # fitted mu_0 value
  RSD_list[ii-1] = sigma(model_bev)    # RSD for this round of nls
  numiter_list[ii-1] = model_bev$convInfo$finIter    # Number of iterations taken by this round of nls
  
  #print('Bev-Holt: RSD = ', signif(sigma(model_bev), 3), ', DOF = ', summary(model_logistic)$df[2], '\n', bev_holt_text)
}

# Put all attempts into dataframe
bev_holt_models = data.frame(a_list, b_list, RSD_list, numiter_list)

# Plot num. NLS iterations vs data point index
plot(bev_holt_models$numiter_list,
     main='NLS iterations vs. index for guessing mu_0 via numerical derivative',
     xlab='Index of data point for numerical derivative',
     ylab='No. of iterations')

# Optimal numerical derivative index for NLS
optimal_index = which(bev_holt_models$numiter_list==min(bev_holt_models$numiter_list[2:iter]))

print_index = paste('Optimal numerical derivative index for NLS convergence = ', optimal_index, sep='')

print(print_index)

