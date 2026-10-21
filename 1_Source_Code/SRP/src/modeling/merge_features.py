import pandas as pd
import os


#cache_dir = 'XXXyour path/Dataset/Dataset001_PDL1Expression/cache/final_features/'
#output_path = 'XXXyour path/Dataset/Dataset001_PDL1Expression/sparad_features_all_patients.csv'
cache_dir = 'XXXyour path/Dataset/Dataset002_ECResponse/cache/final_features/'
output_path = 'XXXyour path/Dataset/Dataset002_ECResponse/sparad_features_all_patients.csv'
cache_lists = os.listdir(cache_dir)
all_results = []
for cache_list in cache_lists:
    cache_file = os.path.join(cache_dir, cache_list)
    df = pd.read_csv(cache_file)
    all_results.append(df)
df_all = pd.concat(all_results, ignore_index=True)
df_all.to_csv(output_path, index=False)




import pandas as pd

# Read the original CSV
#df = pd.read_csv('XXXyour path/Dataset/Dataset001_PDL1Expression/sparad_features_optimized.csv')  # Use '\t' for tab-separated, ',' for comma-separated
df = pd.read_csv('XXXyour path/Dataset/Dataset002_ECResponse/sparad_features_all_patients.csv')
# Pivot the table from long to wide format
df_wide = df.pivot(index='subject_id', 
                   columns='feature_name', 
                   values='feature_value')

df_wide = df_wide.reset_index()
df_wide = df_wide.rename(columns={'subject_id': 'CaseID'})

# Save to CSV
df_wide.to_csv('XXXyour path/Dataset/Dataset002_ECResponse/sparad_features.csv', index=False)