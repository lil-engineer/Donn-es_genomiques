# Project README

## Overview

This project involves filtering and processing genetic data to analyze the distribution of individuals based on Y chromosome SNPs and associated haplogroups and regions. The tasks include filtering SNPs for the Y chromosome, managing missing data, extracting haplogroups and regions, and visualizing the distribution of individuals across different regions.

## Work Assigned

1. **Filter Y Chromosome SNPs**:
   - Read the `.snp` file and filter SNPs corresponding to the Y chromosome.
   - Use `.geno` files to associate filtered SNPs with individuals.

2. **Filter Individuals and Manage Missing Data**:
   - Use `.ind` files to identify and filter duplicates, keeping individuals with the most known SNPs.
   - Eliminate individuals for whom more than 40% of information is missing.

3. **Haplogroup and Region Extraction**:
   - From the individual information files, create a `chrY_info` file by extracting haplogroups and regions corresponding to the individuals from the Y chromosome genome file.

## Work Completed

### 1. Cleaning Files

- **.ind Files**: Cleaned the `.ind` files (`v54.1.p1_HO_public`, `v54.1.p1_HO_public (2)`) to retain relevant columns.
- **.geno File**: Processed the `.geno` file in chunks to manage memory effectively.

```python
import numpy as np
from genetics import loadRawGenoFile

def process_with_memmap_in_chunks(filename, chunk_size=10000):
    geno_file, nind, nsnp, rlen = loadRawGenoFile(filename)
    geno = np.memmap(filename, dtype='uint8', mode='r', shape=(nsnp, rlen))
    for start in range(0, nsnp, chunk_size):
        end = min(start + chunk_size, nsnp)
        chunk = geno[start:end]
        chunk_unpacked = np.unpackbits(chunk, axis=1)[:, :(2 * nind)]
        print(f"Processed chunk {start} to {end} with shape {chunk_unpacked.shape}")

# Example usage
process_with_memmap_in_chunks(r'path_to_file/v54.1.p1_HO_public.geno')
```
### 2. Filtering SNPs

Filtered SNPs corresponding to the Y chromosome from the `.geno` file. (Our .geno file uses SNP IDs or positions, we need to map these to the indices used in the .geno file. This will allow us to select the appropriate rows from the .geno file.)

```python
import pandas as pd
import numpy as np
from genetics import loadRawGenoFile

def load_snp_table(snp_table_filename):
    """Load the SNP table from a CSV file."""
    snp_table = pd.read_csv(snp_table_filename)
    return snp_table

def filter_y_chromosome_snps(geno_filename, snp_table_filename, chunk_size=10000):
    """Filter SNPs corresponding to the Y chromosome from the .geno file."""
    snp_table = load_snp_table(snp_table_filename)
    
    # Filter for Y chromosome SNPs
    y_chromosome_snps = snp_table[snp_table['Chromosome'] == 24]['SNP']
    
    # Map SNP IDs to indices
    snp_id_to_index = {snp_id: idx for idx, snp_id in enumerate(snp_table['SNP'])}
    y_snp_indices = [snp_id_to_index[snp_id] for snp_id in y_chromosome_snps if snp_id in snp_id_to_index]
    
    # Load the .geno file
    geno_file, nind, nsnp, rlen = loadRawGenoFile(geno_filename)
    geno = np.memmap(geno_filename, dtype='uint8', mode='r', shape=(nsnp, rlen))
    
    filtered_data = []
    
    # Process the geno file in chunks
    for start in range(0, nsnp, chunk_size):
        end = min(start + chunk_size, nsnp)
        chunk = geno[start:end]
        chunk_unpacked = np.unpackbits(chunk, axis=1)[:, :(2 * nind)]
        
        # Find Y chromosome SNPs in the current chunk
        chunk_indices = np.array([idx for idx in y_snp_indices if start <= idx < end])
        if chunk_indices.size > 0:
            chunk_filtered = chunk_unpacked[chunk_indices - start]
            filtered_data.append(chunk_filtered)
    
    return np.vstack(filtered_data) if filtered_data else np.array([])

# Example usage
filtered_y_chromosome_data = filter_y_chromosome_snps(
    r'path_to_file/v54.1.p1_HO_public.geno',
    r'path_to_file/v54.1.p1_HO_public (2).csv'
)
print(f'Filtered Y chromosome data shape: {filtered_y_chromosome_data.shape}')
```
### 3. Managing Missing Values

Filtered the individual data to get only relevant columns and then identified and filtered duplicates, keeping individuals with the most known SNPs and eliminating those with more than 40% missing information.

#### Step 1: Filtering Relevant Columns

Filtered the individual data to include only relevant columns.

```python
def create_filtered_ind_df(ind_data):
    """Select only the relevant columns."""
    relevant_columns = ['Genetic ID', 'SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)']
    filtered_ind_data = ind_data[relevant_columns]
    return filtered_ind_data

# Create a new DataFrame with only the relevant columns
filtered_ind_data = create_filtered_ind_df(ind_data)
print(filtered_ind_data.head())
```

#### Step 2: Filtering Individuals Based on Missing Data

Used individual data and Y chromosome filtered data from the `.geno` file to filter out individuals with more than 40% missing data.

```python
import numpy as np
import pandas as pd

def filter_individuals_based_on_missing_data(geno_data, ind_data, missing_threshold=0.4):
    """Filter individuals with more than the specified proportion of missing data."""
    # Convert 'SNPs hit on autosomal targets' to numeric
    ind_data['SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)'] = pd.to_numeric(
        ind_data['SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)'], errors='coerce')
    
    # Extract relevant columns
    ind_ids = ind_data['Genetic ID'].values
    snps_hit = ind_data['SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)'].values
    
    # Count the total number of SNPs available
    total_snps = geno_data.shape[0]
    
    # Calculate the proportion of missing SNPs for each individual
    missing_data = 1 - (snps_hit / total_snps)
    
    # Filter out individuals with more than the missing threshold
    valid_individuals_mask = missing_data <= missing_threshold
    
    # Filter ind_data based on valid individuals
    filtered_ind_data = ind_data[valid_individuals_mask]
    valid_individual_ids = filtered_ind_data['Genetic ID'].values
    
    # Create a dictionary for fast lookup
    id_to_index = {id: idx for idx, id in enumerate(ind_ids)}
    
    # Create a boolean mask for geno_data columns
    geno_indices = [id_to_index[id] for id in valid_individual_ids if id in id_to_index]
    valid_individuals_mask = np.zeros(geno_data.shape[1], dtype=bool)
    valid_individuals_mask[geno_indices] = True
    
    # Filter geno_data based on valid individuals
    filtered_geno_data = geno_data[:, valid_individuals_mask]
    
    return filtered_geno_data, filtered_ind_data

# Example usage
filtered_geno_data, filtered_ind_data = filter_individuals_based_on_missing_data(filtered_y_chromosome_data, filtered_ind_data)
print(f'Filtered genotype data shape: {filtered_geno_data.shape}')
print(f'Filtered individual data shape: {filtered_ind_data.shape}')
```
### 4: Haplogroup and Region Extraction

Merged filtered individual data with additional information and extracted relevant haplogroups and regions.

```python
import pandas as pd

# Merge filtered individual data with additional information
df_chrY_info = pd.merge(filtered_ind_data_selected, ind_data_selected, on='Genetic ID', how='left')

# Rename columns for clarity
df_chrY_info = df_chrY_info.rename(columns={
    'Y haplogroup (manual curation in ISOGG format)': 'Y Haplogroup',
    'mtDNA haplogroup if >2x or published': 'mtDNA Haplogroup'
})

# Check for missing values and create a mask to filter out rows with 'n/a'
cols_to_check = ['Y Haplogroup', 'mtDNA Haplogroup']
mask = ~df_chrY_info[cols_to_check].apply(lambda col: col.str.contains('n/a', case=False, na=False)).any(axis=1)

# Apply the mask to filter the DataFrame and reset index
df_chrY_info_final = df_chrY_info[mask]
df_chrY_info_final = df_chrY_info_final.reset_index(drop=True)

# Display the final DataFrame
print(df_chrY_info_final.head())
```
### Visualization

Created a visualization of the distribution of individuals across regions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Select relevant columns for plotting
relevant_cols = ['Genetic ID', 'Political Entity']
df_plot = df_chrY_info_final[relevant_cols]

# Define a color palette with 10 colors
palette = sns.color_palette('husl', n_colors=10)

# Set the style for the plot
sns.set_style(style='dark')

# Create the plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df_plot, 
              x='Political Entity', 
              order=df_plot['Political Entity'].value_counts().head(10).index, 
              hue='Political Entity', 
              palette=palette, 
              legend=False)

# Set the title and labels
plt.title('Distribution of Individuals Across Regions')
plt.xlabel('Region')
plt.ylabel('Number of Individuals')
plt.xticks(rotation=45)
plt.tight_layout()

# Display the plot
plt.show()
```
![Distribution of Individuals Across Regions](Images/Dist_Across_Region.png)
## Conclusion

The project involved several key steps to process and analyze genetic data:

1. **Filtering Y Chromosome SNPs**: We filtered SNPs specific to the Y chromosome from large `.geno` files, effectively managing memory by processing the data in chunks.

2. **Managing Missing Data**: Individual data was filtered to retain relevant columns, and individuals with more than 40% missing data were removed. The code ensured that only those with sufficient SNP information were included.

3. **Haplogroup and Region Extraction**: We merged filtered individual data with additional information to extract relevant haplogroups and regions. This included cleaning data by removing entries with 'n/a' values and resetting the index for a clean dataset.

4. **Visualization**: A visualization was created to illustrate the distribution of individuals across different regions. The plot provided insights into the geographic distribution of the individuals based on the extracted haplogroups and regions.

The provided code snippets demonstrate how each task was accomplished, from data cleaning and filtering to visualization. This approach ensures a comprehensive analysis of the genetic data, with clear insights into the distribution of individuals and the quality of the data.




