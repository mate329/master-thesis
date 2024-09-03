from statsmodels.stats.anova import AnovaRM
import pandas as pd

import pandas as pd

# Load your dataset
data = pd.read_csv('balanced_accelerometer_data.csv')

data['day'] = data['date'].str.split('/').str[0].astype(int)
data = data[data['day'].isin([1, 2, 3])]


print(data.head())



# Perform ANOVA for angularSpeedX
anova_x = AnovaRM(data, 'angularSpeedX', 'username', within=['day'], aggregate_func='mean').fit()

# Perform ANOVA for angularSpeedY
anova_y = AnovaRM(data, 'angularSpeedY', 'username', within=['day'], aggregate_func='mean').fit()

# Perform ANOVA for angularSpeedZ
anova_z = AnovaRM(data, 'angularSpeedZ', 'username', within=['day'], aggregate_func='mean').fit()

anova_x.summary(), anova_y.summary(), anova_z.summary()
