# Importing packages

import numpy as np
import pandas as pd
import csv
import scipy

# Import datasets
areas = pd.read_csv('data/doi_sites/sampling_area.csv')
sites = pd.read_csv('data/doi_sites/sample_site.csv')
frac_cover = pd.read_csv('data/doi_sites/fractional_cover.csv')
lma_meadow = pd.read_csv('data/doi_lma/lma_meadow_area_samples.csv', na_values='NA')
lma_site = pd.read_csv('data/doi_lma/lma_site_samples.csv', na_values='NA')
species = pd.read_csv('data/doi_sites/species_list.csv')
cn = pd.read_csv('data/doi_cn/CN_Results_Foliar.csv')

# Use Bradley and T403 interchangeably
lma_meadow['SamplingArea'].loc[lma_meadow['SamplingArea'] == 'T403'] = "BD"
frac_cover['SamplingArea'].loc[frac_cover['SamplingArea'] == 'T403'] = "BD"

# remove unneeded columns, merge vegetation type, and calculate LWA
frac_cover = pd.merge(frac_cover, species, how="left", on=['CoverCode'])
frac_cover = frac_cover[['SampleSiteCode', 'SamplingArea',
                         'CoverCode', 'FractionalCover', 'Type', 'Family', 'Genus']]

lma_meadow = pd.merge(lma_meadow, species, how="left", on=['CoverCode'])
lma_meadow['LWA_gm2'] = (lma_meadow['Wet_Weight_g']-lma_meadow['Dry_Weight_g']
                         )/(lma_meadow['Area_cm2']/(100*100))
lma_meadow = lma_meadow[['SamplingArea', 'CoverCode',
                         'Genus', 'Family', 'Type', 'LMA_gm2', 'LWC_%', 'LWA_gm2']]
lma_meadow['Type'].loc[lma_meadow['Type'] == 'willow'] = "shrub"

# use site level LMA sheet as base.
lma_site = pd.merge(lma_site, species, how="left", on=['CoverCode'])
lma_site['LWA_gm2'] = (lma_site['Wet_Weight_g']-lma_site['Dry_Weight_g']) / \
    (lma_site['Area_cm2']/(100*100))
lma_site = lma_site[['SamplingArea', 'SampleSiteCode', 'CoverCode',
                     'Genus', 'Family', 'Type', 'LMA_gm2', 'LWC_%', 'LWA_gm2']]

# Now, remove all sites from fractional cover that are in the LMA site data
# and reassign willows that exist in meadows to 'shrub' category

all_cover = frac_cover.loc[frac_cover['SampleSiteCode'].isin(lma_site['SampleSiteCode'])]
frac_cover = frac_cover.loc[~frac_cover['SampleSiteCode'].isin(lma_site['SampleSiteCode'])]
frac_cover['Type'].loc[frac_cover['Type'] == 'willow'] = "shrub"

# For all fractional species in meadow sites that have LMA values from that sampling area, assign local LMA value
all_cover_lma = pd.merge(all_cover, lma_site, how="left", on=[
                         'SamplingArea', 'SampleSiteCode', 'CoverCode', 'Genus', 'Family'])
frac_lma = pd.merge(frac_cover, lma_meadow, how="left", on=[
                    'SamplingArea', 'CoverCode', 'Genus', 'Family', 'Type'])


# Remove bare and litter cover
frac_lma.drop(frac_lma[frac_lma['CoverCode'] == 'Bare'].index, inplace=True)
frac_lma.drop(frac_lma[frac_lma['CoverCode'] == 'Litter'].index, inplace=True)

# generate conditions for data-filling
# 1. Fill with species median LMA from all other sites, then by genus and family.
# 2. For OF and OG, fill with forb or gram median from sampling area

# Combine datasets
all_scans = pd.concat([lma_meadow, lma_site], axis=0)

# Group datasets by species/genus/family
species_lma = all_scans.groupby('CoverCode', as_index=False).median()
genus_lma = all_scans.groupby('Genus', as_index=False).median()
family_lma = all_scans.groupby('Family', as_index=False).median()
type_lma = all_scans.groupby('Type', as_index=False).median()
type_area_lma = all_scans.groupby(['Type', 'SamplingArea'], as_index=False).median()

# replace missing values for species

missing_species = frac_lma["CoverCode"].loc[pd.isna(frac_lma['LWC_%']) & (
    frac_lma['CoverCode'] != 'OF') & (frac_lma['CoverCode'] != 'OG')].unique()
for i in missing_species:
    g = species.loc[species['CoverCode'] == i, 'Genus'].values[0]
    f = species.loc[species['CoverCode'] == i, 'Family'].values[0]
    t = species.loc[species['CoverCode'] == i, 'Type'].values[0]
    replace_rows = pd.isna(frac_lma['LWC_%']) & (frac_lma['CoverCode'] == i)
    if species_lma.loc[species_lma["CoverCode"] == i, ("LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_rows, ("LMA_gm2", "LWC_%", "LWA_gm2")
                     ] = species_lma.loc[species_lma["CoverCode"] == i, ("LMA_gm2", "LWC_%", "LWA_gm2")].values

    # if there are samples of this genus, replace with the average values of those samples
    elif genus_lma.loc[genus_lma["Genus"] == g, ("LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_rows, ("LMA_gm2", "LWC_%", "LWA_gm2")
                     ] = genus_lma.loc[genus_lma["Genus"] == g, ("LMA_gm2", "LWC_%", "LWA_gm2")].values

    # if there are samples of this family, replace with the average of those samples
    elif family_lma.loc[family_lma["Family"] == f, ("LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_rows, ("LMA_gm2", "LWC_%", "LWA_gm2")
                     ] = family_lma.loc[family_lma["Family"] == f, ("LMA_gm2", "LWC_%", "LWA_gm2")].values

    elif type_lma.loc[type_lma["Type"] == t, ("LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_rows, ("LMA_gm2", "LWC_%", "LWA_gm2")
                     ] = type_lma.loc[type_lma["Type"] == t, ("LMA_gm2", "LWC_%", "LWA_gm2")].values

# Backfill for other gram / forb

# sites that need back filling:
backfill_sites = frac_lma["SamplingArea"].loc[
    (frac_lma['CoverCode'] == 'OF') | (frac_lma['CoverCode'] == 'OG')].unique()

for i in backfill_sites:
    replace_of = ((frac_lma['CoverCode'] == 'OF') & (frac_lma['SamplingArea'] == i))
    replace_og = ((frac_lma['CoverCode'] == 'OG') & (frac_lma['SamplingArea'] == i))
    if type_area_lma.loc[(type_area_lma["Type"] == 'forb') & (type_area_lma['SamplingArea'] == i), (
            "LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_of, ("LMA_gm2", "LWC_%", "LWA_gm2")] = type_area_lma.loc[
            (type_area_lma["Type"] == 'forb') & (type_area_lma['SamplingArea'] == i), (
                "LMA_gm2", "LWC_%", "LWA_gm2")].values
    else:
        frac_lma.loc[replace_of, ("LMA_gm2", "LWC_%", "LWA_gm2")] = type_lma.loc[
            (type_lma["Type"] == 'forb'), ("LMA_gm2", "LWC_%", "LWA_gm2")].values

    if type_area_lma.loc[(type_area_lma["Type"] == 'gram') & (type_area_lma['SamplingArea'] == i), (
            "LMA_gm2", "LWC_%", "LWA_gm2")].size > 0:
        frac_lma.loc[replace_og, ("LMA_gm2", "LWC_%", "LWA_gm2")] = type_area_lma.loc[
            (type_area_lma["Type"] == 'gram') & (type_area_lma['SamplingArea'] == i), (
                "LMA_gm2", "LWC_%", "LWA_gm2")].values
    else:
        frac_lma.loc[replace_og, ("LMA_gm2", "LWC_%", "LWA_gm2")] = type_lma.loc[
            (type_lma["Type"] == 'gram'), ("LMA_gm2", "LWC_%", "LWA_gm2")].values

# Output
frac_lma['LMAFrac'] = frac_lma['LMA_gm2'] * frac_lma['FractionalCover']
frac_lma['LWAFrac'] = frac_lma['LWA_gm2'] * frac_lma['FractionalCover']
frac_lma['LWCFrac'] = frac_lma['LWC_%'] * frac_lma['FractionalCover']

agg_lma = frac_lma.groupby('SampleSiteCode', as_index=False)[
    ['LMAFrac', 'LWAFrac', 'LWCFrac', 'FractionalCover']].sum()
agg_lma['LMA_gm2'] = agg_lma['LMAFrac'] / agg_lma['FractionalCover']
agg_lma['LWA_gm2'] = agg_lma['LWAFrac'] / agg_lma['FractionalCover']
agg_lma['LWC_%'] = agg_lma['LWCFrac'] / agg_lma['FractionalCover']
agg_lma = agg_lma[['SampleSiteCode', 'LMA_gm2', 'LWA_gm2', 'LWC_%']]

all_lma = pd.concat([agg_lma, lma_site], axis=0, join='inner')

site_trait_data = sites[['SampleSiteID', 'SampleSiteCode', 'Site_Veg', 'Needles']]
cn_merging = cn[['SampleSiteCode', 'd15N', 'd13C', 'N_weight_percent', 'C_weight_percent']]
cn_merging['CN'] = ((cn_merging['C_weight_percent'] / 100) / 12.0107)/((cn_merging['N_weight_percent'] / 100) / 14.0067)

site_trait_data = pd.merge(site_trait_data, all_lma, on='SampleSiteCode', how='left')
site_trait_data = pd.merge(site_trait_data, cn_merging, on='SampleSiteCode', how='left')

# export dataset
site_trait_data.to_csv('data/site_trait_data.csv', index=False)
