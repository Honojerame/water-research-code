import pandas as pd

# Load the aligned data
aligned_path = r'c:\Water_research\modeling\aligned_modeling_data.csv'
aligned = pd.read_csv(aligned_path, dtype={'site_number': str})

print(f"Original shape: {aligned.shape}")
print(f"Original columns: {list(aligned.columns)}")

# Remove the empty columns (srad, vp, swe)
aligned = aligned.drop(columns=['srad', 'vp', 'swe'])

print(f"\nNew shape: {aligned.shape}")
print(f"New columns: {list(aligned.columns)}")

# Save back to the same file
aligned.to_csv(aligned_path, index=False)
print(f"\n✓ Updated file saved: {aligned_path}")
