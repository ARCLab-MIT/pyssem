import glob
import os
import pandas as pd

def alter_phase_for_debris(df, txt_file):
    """
    Update phase to 4 for all obj_ids that match IDs in the given text file.
    """
    # Ensure ID column exists
    if 'obj_id' not in df.columns:
        raise ValueError("Expected column 'obj_id' not found in DataFrame.")

    # Convert to 15-digit zero-padded strings
    df['obj_id_str'] = df['obj_id'].astype(int).astype(str).str.zfill(18)

    # Load and clean the ID list from file
    with open(txt_file, 'r') as f:
        id_list = [line.strip() for line in f if line.strip()]

    # Match and update
    mask = df['obj_id_str'].isin(id_list)
    df.loc[mask, 'phase'] = 4

    print(f"Updated {mask.sum()} entries to phase 4 (debris).")
    return df

# 000000000000100 <- example df
# 000000000000100001 <- example text file

def assign_species(df):

    txt_file = 'misclassified_debris.txt'

    df = alter_phase_for_debris(df, txt_file)
    
    for col in ['obj_type', 'phase', 'maneuverable', 'mass']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['species_class'] = pd.NA

    # Unslotted Satellites: 'Su'
    mask = (
        (df['obj_type']     == 2) &
        (df['phase']        == 2) &
        (df['maneuverable'] == 1)
    )
    df.loc[mask, 'species_class'] = 'Su'

    # Cubesats: 'Sns'  (mass ≤ 20 kg)
    mask = (
        (df['obj_type']     == 2) &
        (df['phase']        == 2) &
        (df['mass']         <= 20)
    )
    df.loc[mask, 'species_class'] = 'Sns'

    # Slotted Satellites: 'S'
    mask = (
        (df['obj_type'] == 2) &
        (df['phase']    == 2) &
        df['const_name'].notna()
    )
    df.loc[mask, 'species_class'] = 'S'

    # Debris: 'N'
    mask = df['obj_type'] >= 3
    mask = (
        (df['obj_type'] >= 3) &
        (df['phase']    == 4)
    )
    df.loc[mask, 'species_class'] = 'N'

    # Rocket Bodies: 'B'
    mask = df['obj_type'] == 1
    df.loc[mask, 'species_class'] = 'B'

    # --- Print counts for every category, including NA ---
    counts = df['species_class'].value_counts(dropna=False)
    print("Species counts (including NA) :")
    for cls, cnt in counts.items():
        print(f"  {cls!s}: {cnt}")

    return df


if __name__ == '__main__':
    import os

    # switch CWD to the folder this script file lives in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Find all CSV files in current directory
    csv_files = glob.glob('*.csv')

    print(csv_files)

    for csv_file in csv_files:
        if 'ref' not in csv_file:
            print(f"Skipping {csv_file} as not a reference scenario")
            continue

        print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        df = assign_species(df)
        df.to_csv(csv_file, index=False)
        print(f"Saved updated file: {csv_file}\n")
