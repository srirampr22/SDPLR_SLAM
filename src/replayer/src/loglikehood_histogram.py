import numpy as np
import matplotlib.pyplot as plt

# Load the log-likelihoods from the CSV file
likelihoods = np.loadtxt("log_likelihoods.csv", delimiter=",")

# Plot the histogram
plt.hist(likelihoods, bins=50)
plt.xlabel('Log-Likelihood')
plt.ylabel('Frequency')
plt.title('Histogram of Log-Likelihoods')
plt.show()
