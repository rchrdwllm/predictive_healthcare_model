import pandas as pd
from disease_preprocess import count_cases
from disease_outbreak import detect_outbreak_per_day, predict_future_outbreaks

# let's say na lang na may master list na tayo ng entry ng users of their symptoms and kasama na dun yung predicted disease
# bale manggagaling 'to sa user input ng symptoms nila
df = pd.read_csv('dataset/sample_user_data.csv')

# for every entry, ippredict yung disease, then isasave sa list sa gantong format ng csv
disease_counts = count_cases(df)

print('Disease counts:\n', disease_counts)

# then let's say overtime malaki na yung dataset natin, pag naglalagay sila ng bagong entry, pede na
# tayo gumawa ng detection ng outbreaks, based sa average cases per prognosis per day
# (or not necessarily namang malaki na yung dataset, basta may historical data tayo ng mga cases at
# naga-add sila ng entries ganun)
detected_outbreaks = detect_outbreak_per_day(disease_counts)

print('\nDetected outbreaks:\n', detected_outbreaks)

# then pede na rin gumawa ng forecast ng disease counts and outbreaks based sa historical data
# let's say, for the next 7 days
forecasted_outbreaks = predict_future_outbreaks(disease_counts, days=7)

print('\nForecasted outbreaks:\n', forecasted_outbreaks)
