library(mice)
# Read CSV file that exported from Python
local <- read.csv(file = './data/clean/local_2013_9_12.csv')

# Impute missing values using mice
local_imputed <- mice(data=local, m=5, method="cart", maxit=20, where = is.na(local))
# Pick one of the inerations to be the complete data
local_completeData <- complete(local_imputed, 5)

# Check how well mice imputed NA's
sum(is.na(x.completeData))
## All NAs are in Invetory_FP
# Replace NAs with 0
local_completeData[is.na(local_completeData)] <- 0

# Export mice imputation result to CSV
write.csv(local_completeData,'./data/clean/local_mice.csv', row.names = FALSE)