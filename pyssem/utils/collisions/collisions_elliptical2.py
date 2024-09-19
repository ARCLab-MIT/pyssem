from utils.collisions.collisions import *
from utils.collisions.collisions_elliptical import func_de


def create_collision_pairs(scen_properties):
    """
    Function takes a scen_properties object with a list of species and the same species organised in species_cells into 
    archetypical categories. It calculate and creates a set of species_pair objects which is stored at the scen_properties
    object and used to compile collision equations during the model building process.

    The model is aware of trackability, maneuverability and slotting. Object fragmentation counts are based on the NASA 
    Standard Breakup model. 

    :param scen_properties: ScenarioProperties object
    :type scen_properties: ScenarioProperties
    """
    
    # Get the binomial coefficient of the species
    # This returns all possible combinations of the species
    species =  [species for species_group in scen_properties.species.values() for species in species_group]
    species_cross_pairs = list(combinations(species, 2))
    species_self_pairs = [(s, s) for s in species]

    # Combine the cross and self pairs
    species_pairs = species_cross_pairs + species_self_pairs
    species_pairs_classes = [] 
    # n_f = symbols('n_f:{0}'.format(scen_properties.n_shells))

    # Debris species - remember, we don't want PMD linked species. Just raw debris.
    # debris_species = [species for species in scen_properties.species['debris'] if not species.pmd_linked_species]
    debris_species = [species for species in scen_properties.species['debris']]

    # Calculate the Mass bin centres, edges and widths
    binC = np.zeros(len(debris_species))
    binE_mass = np.zeros(2 * len(debris_species))
    binW = np.zeros(len(debris_species))
    LBgiven = scen_properties.LC

    for index, debris in enumerate(debris_species):
        binC[index] = debris.mass
        binE_mass[2 * index: 2 * index + 2] = [debris.mass_lb, debris.mass_ub]
        binW[index] = debris.mass_ub - debris.mass_lb

    binE_mass = np.unique(binE_mass)

    # Eccentricity Binning, multiple debris species will have the same eccentricity bins
    binE_ecc = debris_species[0].eccentricity_bins
    binE_ecc = np.insert(binE_ecc, 0, 0) # Insert 0 at the beginning
    binE_ecc = np.append(binE_ecc, 1) # Append 1 at end 

    args = [(i, species_pair, scen_properties, debris_species, binE_mass, binE_ecc, LBgiven) for i, species_pair in enumerate(species_pairs)]
    
    # Use multiprocessing Pool for parallel processing
    if scen_properties.parallel_processing:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_species_pair, args), total=len(species_pairs), desc="Creating collision pairs"))
    else:
        results = [process_species_pair(arg) for arg in tqdm(args, desc="Creating collision pairs")]

    # Collect results
    species_pairs_classes.extend(results)

    return species_pairs_classes

def process_species_pair(args):
    
    i, (s1, s2), scen_properties, debris_species, binE_mass, binE_ecc, LBgiven = args
    m1, m2 = s1.mass, s2.mass
    r1, r2 = s1.radius, s2.radius

    # print(f"{s1.sym_name} vs. {s2.sym_name}")

    # Create a matrix of gammas, rows are the shells, columns are debris species (only 2 as in loop)
    gammas = Matrix(scen_properties.n_shells, 2, lambda i, j: -1)

    # Create a list of source sinks, first two are the active species
    source_sinks = [s1, s2]

    # Implementing logic for gammas calculations based on species properties
    if s1.maneuverable and s2.maneuverable:
        # Multiplying each element in the first column of gammas by the product of alpha_active values
        gammas[:, 0] = gammas[:, 0] * s1.alpha_active * s2.alpha_active
        if s1.slotted and s2.slotted:
            # Applying the minimum slotting effectiveness if both are slotted
            gammas[:, 0] = gammas[:, 0] * min(s1.slotting_effectiveness, s2.slotting_effectiveness)

    elif (s1.maneuverable and not s2.maneuverable) or (s2.maneuverable and not s1.maneuverable):
        if s1.trackable and s2.maneuverable:
            gammas[:, 0] = gammas[:, 0] * s2.alpha
        elif s2.trackable and s1.maneuverable:
            gammas[:, 0] = gammas[:, 0] * s1.alpha

    # Applying symmetric loss to both colliding species
    gammas[:, 1] = gammas[:, 0]

    # Rocket Body Flag - 1: RB; 0: not RB
    # Will be 0 if both are None type
    if s1.RBflag is None and s2.RBflag is None:
        RBflag = 0
    elif s1.RBflag is None:
        RBflag = s2.RBflag
    elif s2.RBflag is None:
        RBflag = s1.RBflag
    else:
        RBflag = max(s1.RBflag, s2.RBflag)

    # Calculate the number of fragments made for each debris species
    frags_made = np.zeros((len(scen_properties.v_imp2), len(debris_species)))
    alt_nums = np.zeros((scen_properties.n_shells * 2, len(debris_species)))

    # This will tell you the number of fragments in each debris bin
    for dv_index, dv in enumerate(scen_properties.v_imp2): # This is the case for circular orbits 

        #v1, dv2 = 10, 10 # for now we are going to assume the same velocity. This can change later. 
        v1 = s1.velocity_per_shells[dv_index][dv_index]
        v2 = s2.velocity_per_shells[dv_index][dv_index]

        try:
            results = evolve_bins_de(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, dv_index, n_shells=0)

            # if s1.elliptical or s2.elliptical:
            #     if s1.elliptical and s2.elliptical:
            #         # Both are elliptical, take the product of the time_per_shells values
            #         time_factor = s1.time_per_shells[dv_index][dv_index] * s2.time_per_shells[dv_index][dv_index]
            #     elif s1.elliptical:
            #         time_factor = s1.time_per_shells[dv_index][dv_index]
            #     else:
            #         # Only s2 is elliptical, use its time_per_shells value
            #         time_factor = s2.time_per_shells[dv_index][dv_index]
                
            #     frags_made[dv_index, :] = results[0] * time_factor
            #     alt_nums = results[3] * time_factor

            # else:
            #     frags_made[dv_index, :] = results[0]
            #     alt_nums = results[3]
        except IndexError as ie:
            alt_nums  = None
            continue
        except ValueError as e:
            continue

    # for i, species in enumerate(debris_species):
    #     frags_made_sym = Matrix(frags_made[:, i]) 

    #     # Multiply it by the likelihood of collision (gammas) to get the number of fragments made for each shell
    #     new_column = -gammas[:, 1].multiply_elementwise(frags_made_sym)
    #     new_column = new_column.reshape(gammas.rows, 1)  # Ensure it's a column vector

    #     # Use col_insert to add the new column. Insert at index 2+i
    #     gammas = gammas.col_insert(2 + i, new_column)

    #     if 2 + i < len(source_sinks):
    #         source_sinks[2 + i] = species
    #     else:
    #         source_sinks.append(species)

    
    return SpeciesPairClass(s1, s2, gammas, source_sinks, scen_properties, alt_nums)

def evolve_bins_de(scen_properties, m1, m2, r1, r2, v1, v2, binE_mass, binE_ecc, collision_index, n_shells=0):
    SS = 20  # super sampling ratio
    MU = 398600.4418  # km^3/s^2
    RE = 6378.1  # km

    # Ensure that m1 > m2, if not swap
    if m1 < m2:
        m1, m2 = m2, m1
        r1, r2 = r2, r1

    dv = v1
    catastrophe_ratio = (m2 * ((dv * 1000) ** 2) / (2 * m1 * 1000))  # J/g = kg*(km/s)^2 / g

    # If the specific energy is < 40 J/g: non catastrophic collision
    if catastrophe_ratio < 40:
        M = m2 * dv ** 2  # Correction from ODQN [kg*km^2/s^2]
        isCatastrophic = 0
    else:  # Catastrophic collision
        M = m1 + m2
        isCatastrophic = 1

    # find the number of fragments generated
    num = (0.1 * M ** 0.75 * scen_properties.LC ** (-1.71)) - (0.1 * M ** 0.75 * min(1, 2 * r1) ** (-1.71))
    numSS = SS * num

    # Create PDF of power law distribution, then sample 'num' selections to find each mass
    dd_edges = np.logspace(np.log10(scen_properties.LC), np.log10(min(1, 2 * r1)), 500)
    dd_means = 10 ** (np.log10(dd_edges[:-1]) + np.diff(np.log10(dd_edges)) / 2)

    # Cumulative distribution
    nddcdf = 0.1 * M ** 0.75 * dd_edges ** (-1.71)
    ndd = np.maximum(0, -np.diff(nddcdf))

    # Make sure int, 0 to 1, random number for stochastic sampling of fragment diameters
    repeat_counts = np.floor(ndd).astype(int) + (np.random.rand(len(ndd)) > (1 - (ndd - np.floor(ndd)))).astype(int)
    d_pdf = np.repeat(dd_means, repeat_counts)

    try:
        dss = d_pdf[np.random.randint(0, len(d_pdf), size=int(np.ceil(numSS)))]
    except ValueError:
        dss = 0

    # Calculate the mass of objects
    A = 0.556945 * dss ** 2.0047077  # Equation 2.72
    Am = func_Am(dss, 5 if isCatastrophic == 0 else 0)  # use Am conversion of the larger object

    # order the output in descending order
    Am = np.sort(Am)[::-1]
    m = A / Am

    # Remove any fragments that are more than the original mass of m1
    m = m[m < m1]

    if max(m) == 0:
        # No fragments bigger the LNT are generated
        return None

    # Find difference in orbital velocity for shells
    dDV = np.abs(np.median(np.diff(np.sqrt(MU / (RE + scen_properties.HMid)) * 1000)))

    # Compute delta-v for the fragments
    dv_values = func_dv(Am, 'col') / 1000  # km/s

    # Generate random directions for the velocity vectors
    u = np.random.rand(len(dv_values)) * 2 - 1
    theta = np.random.rand(len(dv_values)) * 2 * np.pi
    v = np.sqrt(1 - u ** 2)
    p = np.vstack((v * np.cos(theta), v * np.sin(theta), u)).T
    dv_vec = p * dv_values[:, np.newaxis]  # velocity vectors

    # Calculate the new eccentricity of the fragments
    a = scen_properties.HMid[collision_index] + scen_properties.re
    r_vec = np.tile([a, 0, 0], (len(m), 1))  # create an R
    v_vec = [np.linalg.norm(vec) for vec in dv_vec] + v1  # create a v vec
    v_vec_2d = np.array([[0, v, 0] for v in v_vec])
    new_ecc = func_de(r_vec, v_vec_2d)

    # create random values between 0 and w, length of m
    #new_ecc = np.random.rand(len(m)) * 2 - 1



    # dv = np.sqrt(dv_vec[:, 0] ** 2 + dv_vec[:, 1] ** 2 + dv_vec[:, 2] ** 2)  # Magnitude of velocity

    # Step 2: Construct the data array for np.histogramdd
    # This needs to be a 2D array where each row is [dv, mass, eccentricity] for each fragment
    data = np.array([dv_vec.ravel()[:len(m)], m, new_ecc]).T  # Transpose to get the right shape (n_samples, 3)

    # Step 3: Define the bins for the 3D histogram
    bins = [np.arange(-n_shells, n_shells) * dDV / 1000, binE_mass, binE_ecc]

    # Step 4: Use np.histogramdd to create the 3D histogram
    hist, edges = np.histogramdd(data, bins=bins)

    hist = hist/SS

    # Sum over the velocity bins to collapse the 3D histogram into a 2D histogram
    mass_ecc_slice = np.sum(hist, axis=0)

    # if len(m) > 1000:
    #     # Create a 2D heatmap for mass vs eccentricity
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(mass_ecc_slice.T, origin='lower', aspect='auto',
    #                extent=[edges[1][0], edges[1][-1], edges[2][0], edges[2][-1]])
    #     plt.colorbar(label='Count')
    #     plt.xlabel('Mass (mass bins)')
    #     plt.ylabel('Eccentricity (eccentricity bins)')
    #     plt.title(f'{collision_index} Mass vs Eccentricity (Summed over Velocity) {m1} vs {m2}')
    #     plt.savefig(f'figures/elliptical/mass_eccentricity_{m1}_{m2}.png')
    #     plt.close()

    return hist
