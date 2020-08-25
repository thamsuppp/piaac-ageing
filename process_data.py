### File with all the helper functions to process the PIAAC Data. For more info look at 'PIAAC Data Processing Multiple Countries.ipynb

import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW
import os

### MAPPINGS TO RECODE VARIABLES

# Industry mappings
ISIC1C_mapping = {
    'A': 'Agriculture, forestry and fishing',
    'B': 'Mining and quarrying',
    'C': 'Manufacturing',
    'D': 'Electricity, gas, steam and air conditioning supply',
    'E': 'Water supply; sewerage, waste management and remediation activities',
    'F': 'Construction',
    'G': 'Wholesale and retail trade; repair of motor vehicles and motorcycles',
    'H': 'Transportation and storage',
    'I': 'Accommodation and food service activities',
    'J': 'Information and communication',
    'K': 'Financial and insurance activities',
    'L': 'Real estate activities',
    'M': 'Professional, scientific and technical activities',
    'N': 'Administrative and support service activities',
    'O': 'Public administration and defence; compulsory social security',
    'P': 'Education',
    'Q': 'Human health and social work activities',
    'R': 'Arts, entertainment and recreation',
    'S': 'Other service activities',
    'T': 'Activities of households as employers',
    'U': 'Activities of extraterritorial organizations and bodies',
    '9995': 'No paid work for last 5 years',
    '9996': 'No data',
    '9997': 'No data',
    9997: 'No data',
    '9998': 'No data',
    9998: 'No data',
    '9999': 'No data'
}

# Occupation mappings
ISCO1C_mapping = {
    '0': 'Armed forces',
    0: 'Armed forces',
    '1': 'Legislators, senior officials and managers',
    1: 'Legislators, senior officials and managers',
    '2': 'Professionals',
    2: 'Professionals',
    '3': 'Technicians and associate professionals',
    3: 'Technicians and associate professionals',
    '4': 'Clerks',
    4: 'Clerks',
    '5': 'Service workers and shop and market sales workers',
    5: 'Service workers and shop and market sales workers',
    '6': 'Skilled agricultural and fishery workers',
    6: 'Skilled agricultural and fishery workers',
    '7': 'Craft and related trades workers',
    7: 'Craft and related trades workers',
    8: 'Plant and machine operators and assemblers',
    '8': 'Plant and machine operators and assemblers',
    '9': 'Elementary occupations',
     9: 'Elementary occupations',
    '9995': 'No paid work for last 5 years',
    9995: 'No paid work for last 5 years',
    '9996': 'No data',
    9996: 'No data',
    '9997': 'No data',
    9997: 'No data',
    '9998': 'No data',
    9998: 'No data',
    '9999': 'No data',
    9999: 'No data'
}

EMPLOYMENT_STATUS_mapping = {
    '1': 'Employed', '2': 'Unemployed', '3': 'Out of Labor Force', '4': 'Not known',
    1: 'Employed', 2: 'Unemployed', 3: 'Out of Labor Force', 4: 'Not known'
}

# Mapping PIAAC's age bin to age
AGEGELFS_mapping = {
    1: 17.5, 2: 22.5, 3: 27.5, 4: 32.5, 5: 37.5, 6: 42.5, 7: 47.5, 8: 52.5, 9: 57.5, 10: 62.5
}

SECTOR_mapping = {
    '1': 'Private', '2': 'Public', '3': 'Non-profit'
}

BIZSIZE_mapping = {
    '1': '1 to 10', '2': '11 to 50', '3': '51 to 250', '4': '251 to 1000', '5': 'More than 1000'
}

CONTRACT_mapping = {
    '1': 'Indefinite contract', '2': 'Fixed term contract', '3': 'Temporary contract', '3': 'Apprenticeship', 
    '5': 'No Contract', '6': 'Other'
}

AGESTARTWORKINGFOREMPLOYER_CAT_mapping = {
    '1': 17.5, '2': 22.5, '3': 27.5, '4': 32.5, '5': 37.5, '6': 42.5, '7': 47.5, '8': 52.5, '9': 57.5
}

GENDER_mapping = {
    1: 'Male', 2: 'Female'
}


# Condensed industry mappings
industry_condensed_mapping = {
    'Financial and insurance activities': 'Finance & Insurance',
    'Manufacturing': 'Manufacturing',
    'Construction': 'Construction',
    
    'Water supply; sewerage, waste management and remediation activities': 'Utilities & other goods',
    'Agriculture, forestry and fishing': 'Utilities & other goods',
    'Mining and quarrying': 'Utilities & other goods',
    
    'Wholesale and retail trade; repair of motor vehicles and motorcycles': 'Wholesale & retail',

    'Transportation and storage': 'Transportation & storage',
    'Accommodation and food service activities': 'Accommodation & food service',
    'Information and communication': 'Information & communication',

    'Real estate activities': 'Business services',
    'Professional, scientific and technical activities': 'Business services',

    'Education': 'Other services',
    'Human health and social work activities': 'Other services',
    'Other service activities': 'Other services',
    'Activities of households as employers': 'Other services',
    'Public administration and defence; compulsory social security': 'Other services',
    'Activities of extraterritorial organizations and bodies': 'Other services',
    'No paid work for last 5 years': np.nan
}

# Map occupations to PMET vs non-PMETs
occupation_condensed_mapping = {
    'Legislators, senior officials and managers': 'PMET',
    'Professionals': 'PMET',
    'Technicians and associate professionals': 'PMET',
    'Service workers and shop and market sales workers': 'Non-PMET',
    'Clerks': 'Non-PMET',
    'Plant and machine operators and assemblers': 'Non-PMET',
    'Elementary occupations': 'Non-PMET',
    'Craft and related trades workers': 'Non-PMET',
    'Armed forces': 'Non-PMET',
    'No paid work for last 5 years': np.nan
}

# Mapping granular skill questions
skill_mappings = {'F_Q01b': 'Cooperating with co-workers',
                 'F_Q02a': 'Influence: Sharing work-related info',
                 'F_Q02b': 'Influence: Teaching people',
                 'F_Q02c': 'Influence: Presentations',
                 'F_Q02d': 'Influence: Selling',
                 'F_Q02e': 'Influence: Advising people',
                 'F_Q04a': 'Influence: Influencing people',
                 'F_Q04b': 'Influence: Negotiating with people',
                 'F_Q03a': 'Planning: Planning own activities',
                 'F_Q03b': 'Planning: Planning others activities',
                 'F_Q03c': 'Planning: Organizing own time',
                 'F_Q05a': 'Problem Solving: Simple problems',
                 'F_Q05b': 'Problem Solving: Complex problems',
                 'F_Q06b': 'Physical: Working physically for long',
                 'F_Q06c': 'Physical: Using hands or fingers',
                 'G_Q01a': 'Reading: Reading directions or instructions',
                 'G_Q01b': 'Reading: Reading letters, memos or emails',
                 'G_Q01c': 'Reading: Reading newspapers or magazines',
                 'G_Q01d': 'Reading: Reading journals or publications',
                 'G_Q01e': 'Reading: Reading books',
                 'G_Q01f': 'Reading: Reading manuals or reference materials',
                 'G_Q01g': 'Reading: Reading financial statements',
                 'G_Q01h': 'Reading: Reading diagrams, maps or shematics',
                 'G_Q02a': 'Writing: Writing letters, memos or emails',
                 'G_Q02b': 'Writing: Writing articles',
                 'G_Q02c': 'Writing: Writing reports',
                 'G_Q02d': 'Writing: Fill in forms',
                 'G_Q03b': 'Numeracy: Calculating costs or budgets',
                 'G_Q03c': 'Numeracy: Calculate fractions or %',
                 'G_Q03d': 'Numeracy: Use calculator',
                 'G_Q03f': 'Numeracy: Prepare charts, graphs or tables',
                 'G_Q03g': 'Numeracy: Use simple algebra or formulas',
                 'G_Q03h': 'Numeracy: Use advanced math or statistics',
                 'G_Q05a': 'Internet: Mail',
                 'G_Q05c': 'Internet: Work related info',
                 'G_Q05d': 'Internet: Conduct transactions',
                 'G_Q05e': 'Computer: Spreadsheets',
                 'G_Q05f': 'Computer: Word',
                 'G_Q05g': 'Computer: Programming Language',
                 'G_Q05h': 'Computer: Real-time discussions'}




# Function that returns {n_samples} income samples from the {decile}th decile
def sample_income(decile):
    if np.isnan(decile):
        return np.nan
    sample = np.random.uniform((decile - 1) / 10, decile / 10)
    # Converts the samples into ln income - Same as qnorm in R, loc is mean, scale is sd
    ln_income_sample = stats.norm.ppf(sample, loc = 8.31006, scale = 0.68806)
    income_sample = np.exp(ln_income_sample)
    return income_sample

# From income decile, imputes income
def impute_income(df):
    df['INCOME'] = df['INCOME_DECILE'].apply(lambda x: sample_income(x))
    return df

def impute_age(df):
    
    df2 = df.copy()
    df_age = df2[['AGESTARTWORKINGFOREMPLOYER_CAT', 'YEARSTARTWORKINGFOREMPLOYER', 'AGE_BIN']]
    df_age = df_age.dropna()

    df_age['min_age'] = df_age['AGE_BIN'] - 2.5
    df_age['max_age'] = df_age['AGE_BIN'] + 2.5

    df_age['AGESTARTWORKINGFOREMPLOYER_CAT'] = df_age['AGESTARTWORKINGFOREMPLOYER_CAT'].astype(float)
    df_age['min_age_start_working'] = df_age['AGESTARTWORKINGFOREMPLOYER_CAT'] - 2.5
    df_age['max_age_start_working'] = df_age['AGESTARTWORKINGFOREMPLOYER_CAT'] + 2.5
    df_age['years_since_start_working'] = 2015 - df_age['YEARSTARTWORKINGFOREMPLOYER'].astype(int)

    df_age['min_age_2'] = df_age['AGESTARTWORKINGFOREMPLOYER_CAT'] + df_age['years_since_start_working'] - 2.5
    df_age['max_age_2'] = df_age['AGESTARTWORKINGFOREMPLOYER_CAT'] + df_age['years_since_start_working'] + 2.5


    df_age['min_age_new'] = df_age.apply(lambda x: max(x['min_age'], x['min_age_2']), axis = 1)
    df_age['max_age_new'] = df_age.apply(lambda x: min(x['max_age'], x['max_age_2']), axis = 1)
    df_age['estimated_age'] = (df_age['min_age_new'] + df_age['max_age_new']) / 2

    # Merge the estimated age and the rest of the dataset together
    df2 = df2.merge(df_age['estimated_age'], left_index = True, right_index = True , how = 'left')

    df2['AGE'] = df2['estimated_age']
    df2 = df2.drop('estimated_age', axis = 1)
    
    return df2

# Sample from the data 10000 times, with the weights as the probabilities
def bootstrap(df, bootstrap_size = 10000):
    df2 = df.copy()
    df2['WEIGHT'] = df2['WEIGHT'] / df2['WEIGHT'].sum()
    sample_ids = np.random.choice(len(df2), bootstrap_size, p = df2['WEIGHT'])
    sample = df2.iloc[sample_ids, :].reset_index(drop = True)
    # These bootstrapped samples have no more weight - all have equal weight!
    sample = sample.drop('WEIGHT', axis = 1)
    return sample
    
def initial_processing(data, country):

    columns = [ 'GENDER_R', 'YRSQUAL', 'YRSQUAL_T', 'ISIC1C', 'ISIC2C', 'ISCO1C', 'ISCO2C', 'C_D05', 'D_Q03', 'D_Q09',
              'D_Q10', 'D_Q05a1', 'D_Q05a1_C', 'D_Q05a2', 'D_Q06a', 'C_Q09', 'PLANNING', 'INFLUENCE',
               'READWORK', 'TASKDISC', 'WRITWORK', 'ICTWORK_WLE_CA', 'NUMWORK_WLE_CA', 'PLANNING_WLE_CA', 'INFLUENCE_WLE_CA',
               'READWORK_WLE_CA', 'TASKDISC_WLE_CA', 'WRITWORK_WLE_CA', 'AGE_R', 'AGEG5LFS', 'EARNMTHBONUSPPP', 'EARNHRBONUSDCL', 'SPFWT0']

    skill_columns = ['G_Q02a', 'G_Q02b', 'G_Q02c', 'G_Q02d', # Writing
                        'G_Q01a', 'G_Q01b', 'G_Q01c', 'G_Q01d', 'G_Q01e', 'G_Q01f','G_Q01g', 'G_Q01h', # Reading
                        'G_Q03b', 'G_Q03c', 'G_Q03d', 'G_Q03f', 'G_Q03g', 'G_Q03h', # Numeracy
                        'G_Q05a', 'G_Q05c', 'G_Q05d', 'G_Q05e', 'G_Q05f', 'G_Q05g', 'G_Q05h', # ICT - computer
                        'F_Q03a', 'F_Q03b', 'F_Q03c', # Planning
                        'F_Q04a', 'F_Q04b', 'F_Q02a', 'F_Q02b', 'F_Q02c', 'F_Q02d', 'F_Q02e', # Influencing others
                        'F_Q06b', 'F_Q06c', # Physical
                        'F_Q01b', # Cooperating with cowokers
                        'F_Q05a', 'F_Q05b' # Problem solving
                        ] 

    skill_proficiency_columns = ['PVLIT1', 'PVLIT2', 'PVLIT3', 'PVLIT4', 'PVLIT5', 'PVLIT6', 'PVLIT7', 'PVLIT8', 'PVLIT9', 'PVLIT10',
        'PVNUM1', 'PVNUM2', 'PVNUM3', 'PVNUM4', 'PVNUM5', 'PVNUM6','PVNUM7', 'PVNUM8', 'PVNUM9', 'PVNUM10',
        'PVPSL1', 'PVPSL2', 'PVPSL3', 'PVPSL4', 'PVPSL5', 'PVPSL6', 'PVPSL7', 'PVPSL8', 'PVPSL9', 'PVPSL10']

    df = data.loc[:, columns + skill_columns + skill_proficiency_columns]

    # Replace all N or V with na values
    df = df.replace(['N', 'V', 'D', 'R', '.'], np.nan)

    
    rename_columns_dict = {'D_Q03': 'SECTOR', 'ISIC1C': 'INDUSTRY_1D', 'ISIC2C': 'INDSUTRY_2D', 'C_D05': 'EMPLOYMENT_STATUS',
                           'ISCO1C': 'OCCUPATION_1D', 'ISCO2C': 'OCCUPATION_2D',
                           'D_Q09': 'CONTRACTTYPE', 'D_Q05a1_C': 'AGESTARTWORKINGFOREMPLOYER_CAT', 'C_Q09': 'WORK_EXPERIENCE',
                          'D_Q10': 'HRSWORKPERWEEK', 'D_Q05a1': 'AGESTARTWORKINGFOREMPLOYER',
                          'D_Q05a2': 'YEARSTARTWORKINGFOREMPLOYER', 'D_Q06a': 'BIZSIZE',
                          'AGE_R': 'AGE', 'AGEG5LFS': 'AGE_BIN', 'EARNMTHBONUSPPP': 'INCOME', 'EARNHRBONUSDCL': 'INCOME_DECILE', 'SPFWT0': 'WEIGHT'}

    df = df.rename(columns = rename_columns_dict)

    # Changing all the numbers coded as string, to float
    cols_to_convert_to_float = ['INCOME', 'YRSQUAL', 'YRSQUAL_T', 'HRSWORKPERWEEK', 'AGESTARTWORKINGFOREMPLOYER', 'YEARSTARTWORKINGFOREMPLOYER', 'WORK_EXPERIENCE', 
     'PLANNING', 'INFLUENCE', 'READWORK', 'TASKDISC', 'WRITWORK', 'ICTWORK_WLE_CA',
       'NUMWORK_WLE_CA', 'PLANNING_WLE_CA', 'INFLUENCE_WLE_CA', 'READWORK_WLE_CA', 'TASKDISC_WLE_CA', 'WRITWORK_WLE_CA',
     'AGE', 'INCOME_DECILE']
    df[cols_to_convert_to_float] = df[cols_to_convert_to_float].astype(float)
    df[skill_columns] = df[skill_columns].astype(float)
    df[skill_proficiency_columns] = df[skill_proficiency_columns].astype(float)

    # (15 Jul) Process the skill proficiency columns
    df['LITERACY_SCORE'] = df[['PVLIT1', 'PVLIT2', 'PVLIT3', 'PVLIT4', 'PVLIT5', 'PVLIT6', 'PVLIT7', 'PVLIT8', 'PVLIT9', 'PVLIT10']].mean(axis = 1)
    df['NUMERACY_SCORE'] = df[['PVNUM1', 'PVNUM2', 'PVNUM3', 'PVNUM4', 'PVNUM5', 'PVNUM6','PVNUM7', 'PVNUM8', 'PVNUM9', 'PVNUM10']].mean(axis = 1)
    df['DIGITAL_SCORE'] = df[['PVPSL1', 'PVPSL2', 'PVPSL3', 'PVPSL4', 'PVPSL5', 'PVPSL6', 'PVPSL7', 'PVPSL8', 'PVPSL9', 'PVPSL10']].mean(axis = 1)

    # Drop the initial skill proficiency plausible value columns
    df = df.drop(skill_proficiency_columns, axis = 1)

    # Mapping codings to values
    df['INDUSTRY_1D'] = df['INDUSTRY_1D'].replace(ISIC1C_mapping)
    df['OCCUPATION_1D'] = df['OCCUPATION_1D'].astype(object).replace(ISCO1C_mapping)
    df['SECTOR'] = df['SECTOR'].replace(SECTOR_mapping)
    df['AGE_BIN'] = df['AGE_BIN'].replace(AGEGELFS_mapping)
    df['BIZSIZE'] = df['BIZSIZE'].replace(BIZSIZE_mapping)
    df['AGESTARTWORKINGFOREMPLOYER_CAT'] = df['AGESTARTWORKINGFOREMPLOYER_CAT'].replace(AGESTARTWORKINGFOREMPLOYER_CAT_mapping)
    df['CONTRACTTYPE'] = df['CONTRACTTYPE'].replace(CONTRACT_mapping)
    df['GENDER_R'] = df['GENDER_R'].replace(GENDER_mapping)
    df['EMPLOYMENT_STATUS'] = df['EMPLOYMENT_STATUS'].astype(object).replace(EMPLOYMENT_STATUS_mapping)
    df['COUNTRY'] = country

    return df

def get_all_countries_df(is_bootstrap = True):
    countries_to_include = ['bel', 'cze', 'dnk', 'fin', 'fra', 'gbr', 'jpn', 'kor', 'nld', 'nor', 'nzl', 'sgp']
    df_all = pd.DataFrame()
    for country in countries_to_include:

        print(country)
        data = pd.read_csv('Data/prg{}p1.csv'.format(country))
        
        # Subset columns, rename columns, map codings to values, change numbers coded as string, to float
        df = initial_processing(data, country)
        
        if is_bootstrap == True:
            # Bootstrap entire sample
            df = bootstrap(df, bootstrap_size = 100000)
            
        # If Singapore, impute income
        if country == 'sgp':
            df = impute_income(df)
        
        # If New Zealand or Singapore, impute age
        if country == 'nzl' or country == 'sgp':
            df = impute_age(df)

        if df_all.empty:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis = 0)

        print('{} done'.format(country))
    
    df_all = df_all.reset_index(drop = True)
    return df_all

def more_processing(df):

    # Set income of out of labor force and unemployed people to 0
    df['INCOME'] = df.apply(lambda row: 0 if row['EMPLOYMENT_STATUS'] in ['Out of Labor Force', 'Unemployed'] else row['INCOME'], axis = 1)

    # Convert from PPP to USD exchange rate
    ppp = pd.read_csv('ppp_conversion.csv')
    ppp = {country :factor for country, factor in zip(ppp['Country'], ppp['Conversion Factor'])}
    df['INCOME'] = df[['COUNTRY', 'INCOME']].apply(lambda row: row['INCOME'] * ppp[row['COUNTRY']],axis = 1)

    # Impute age start working for employer
    df['AGESTARTWORKINGFOREMPLOYER'] = df.apply(lambda row: row['AGESTARTWORKINGFOREMPLOYER_CAT'] if row['AGESTARTWORKINGFOREMPLOYER'] == np.nan else row['AGESTARTWORKINGFOREMPLOYER_CAT'], axis = 1)
    df = df.drop('AGESTARTWORKINGFOREMPLOYER_CAT', axis = 1)

    # Add year of the survey
    df['YEAR'] = 2011
    df.loc[df['COUNTRY'] == 'sgp', 'YEAR'] = 2014
    df.loc[df['COUNTRY'] == 'nzl', 'YEAR'] = 2014
    
    df = df.replace('No data', np.nan)
    
    # Process age
    df['AGE'] = df.apply(lambda row: row['AGE_BIN'] if np.isnan(row['AGE']) else row['AGE'], axis = 1)
    df['IS_EMPLOYED'] = df['EMPLOYMENT_STATUS'] == 'Employed'
    
    # Correct the gender in strings
    df.loc[df['GENDER_R'] == '2', 'GENDER_R'] = 'Female'
    df.loc[df['GENDER_R'] == '1', 'GENDER_R'] = 'Male'
    
    return df

def skills_occ_industry_processing(df, rename_granular_skills = False):

    # Split data into employed and not employed
    df_not_employed = df.loc[df['EMPLOYMENT_STATUS'] != 'Employed', :]
    df_employed = df.loc[df['EMPLOYMENT_STATUS'] == 'Employed', :]

    # Fill NA values for the ICT skills to be 1 (never)
    df_employed[['G_Q05a', 'G_Q05c', 'G_Q05d', 'G_Q05e', 'G_Q05f', 'G_Q05g', 'G_Q05h']] = \
    df_employed[['G_Q05a', 'G_Q05c', 'G_Q05d', 'G_Q05e', 'G_Q05f', 'G_Q05g', 'G_Q05h']].fillna(1)

    df_employed['ICTWORK_WLE_CA'] = df_employed['ICTWORK_WLE_CA'].fillna(0)

    # Get the average skill use values
    df_employed['WRITING_AVG'] = df_employed[['G_Q02a', 'G_Q02b', 'G_Q02c', 'G_Q02d']].apply(lambda x: x.mean(), axis = 1)
    df_employed['READING_AVG'] = df_employed[['G_Q01a', 'G_Q01b', 'G_Q01c', 'G_Q01d', 
                                              'G_Q01e', 'G_Q01f','G_Q01g', 'G_Q01h']].apply(lambda x: x.mean(), axis = 1)
    df_employed['NUMERACY_AVG'] = df_employed[['G_Q03b', 'G_Q03c', 'G_Q03d', 'G_Q03f', 
                                               'G_Q03g', 'G_Q03h']].apply(lambda x: x.mean(), axis = 1)
    df_employed['ICT_AVG'] = df_employed[['G_Q05a', 'G_Q05c', 'G_Q05d', 'G_Q05e', 'G_Q05f', 
                                          'G_Q05g', 'G_Q05h',]].apply(lambda x: x.mean(), axis = 1)
    df_employed['PLANNING_AVG'] = df_employed[['F_Q03a', 'F_Q03b', 'F_Q03c']].apply(lambda x: x.mean(), axis = 1)
    df_employed['INFLUENCE_AVG'] = df_employed[['F_Q04a', 'F_Q04b', 'F_Q02a', 'F_Q02b', 
                                                'F_Q02c', 'F_Q02d', 'F_Q02e']].apply(lambda x: x.mean(), axis = 1)
    df_employed['PHYSICAL_AVG'] = df_employed[['F_Q06b', 'F_Q06c']].apply(lambda x: x.mean(), axis = 1)

    # Aggregate skills into three mega-skills
    df_employed['KNOWLEDGE_AVG'] = df_employed[['ICT_AVG', 'NUMERACY_AVG', 'READING_AVG', 'WRITING_AVG']].apply(lambda x: x.mean(), axis = 1)
    df_employed['LEADERSHIP_AVG'] = df_employed[['PLANNING_AVG', 'INFLUENCE_AVG']].apply(lambda x: x.mean(), axis = 1)
    
    df_combined = pd.concat([df_employed, df_not_employed], axis = 0)
    
    # Add column for the condensed occupation and industries
    df_combined['OCCUPATION'] = df_combined['OCCUPATION_1D'].replace(occupation_condensed_mapping)
    df_combined['INDUSTRY'] = df_combined['INDUSTRY_1D'].replace(industry_condensed_mapping)
    
    # Rename granular skills
    if rename_granular_skills == True:
        df_combined = df_combined.rename(columns = skill_mappings)
        
    return df_combined