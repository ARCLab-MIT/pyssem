import glob
import os
import pandas as pd

def assign_species(df):
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
        print(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        df = assign_species(df)
        df.to_csv(csv_file, index=False)
        print(f"Saved updated file: {csv_file}\n")
