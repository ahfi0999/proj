import pandas as pd
import numpy as np

np.random.seed(42)
n_new = 20

sources = ['Facebook', 'Google Ads', 'Referral', 'Website', 'Email Campaign']
courses = ['Data Science', 'AI', 'Web Development', 'MBA', 'Cloud Computing']
income_levels = ['Low', 'Medium', 'High']
education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']
location_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
summary_choices = ['Very interested', 'Mildly interested', 'Just exploring', 'Not interested']

new_leads = pd.DataFrame({
    'lead_id': np.arange(2001, 2001 + n_new),
    'source': np.random.choice(sources, n_new),
    'course_interest': np.random.choice(courses, n_new),
    'email_opened': np.random.choice([0, 1], n_new, p=[0.3, 0.7]),
    'clicked_ad': np.random.choice([0, 1], n_new, p=[0.4, 0.6]),
    'income_level1': np.random.choice(income_levels, n_new),
    'education_level': np.random.choice(education_levels, n_new),
    'location_tier': np.random.choice(location_tiers, n_new),
    'days_to_followup': np.random.randint(1, 15, n_new),
    'Summaryofconversation': np.random.choice(summary_choices, n_new)
})

new_leads.to_csv("new_leads.csv", index=False)
print("new_leads.csv created successfully!")
print(new_leads.head())
