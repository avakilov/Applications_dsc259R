# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np

np.set_printoptions(legacy="1.21")


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    # We iterate up to len(ints) - 1 so we don't go out of bounds 
    # when checking the 'next' element
    for i in range(len(ints) - 1):
        # Check if the absolute difference between neighbors is 1
        if abs(ints[i] - ints[i+1]) == 1:
            return True
            
    # If the loop finishes without finding a pair, return False
    return False
    ...


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------

def median_vs_mean(nums):
    # Sort without modifying the original list
    s = sorted(nums)
    
    n = len(s)
    mid = n // 2
    
    # Compute median
    if n % 2 == 1:           # odd length
        median = s[mid]
    else:                    # even length
        median = (s[mid - 1] + s[mid]) / 2
    
    # Compute mean
    mean = sum(s) / n
    
    return median <= mean

    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    result = ""
    
    # build prefixes from n down to 1
    for i in range(n, 0, -1):
        result += s[:i]
        
    return result
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    exploded_lists = []
    
    # Step 1: build exploded ranges
    for x in ints:
        exploded = list(range(x - n, x + n + 1))
        exploded_lists.append(exploded)
    
    # Step 2: compute max width across all numbers
    max_width = 0
    for lst in exploded_lists:
        for num in lst:
            max_width = max(max_width, len(str(num)))
    
    # Step 3: format each exploded list as padded strings
    result = []
    for lst in exploded_lists:
        padded = [str(num).zfill(max_width) for num in lst]
        result.append(" ".join(padded))
    
    return result
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    result = ""
    
    for line in fh:
        line = line.rstrip('\n')   # remove only newline
        if line:                   # avoid errors if the line is empty
            result += line[-1]
            
    return result
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    return A + np.sqrt(np.arange(len(A)))
    ...


def where_square(A):
    roots = np.sqrt(A)
    return roots == np.floor(roots)
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    rows = len(matrix)
    cols = len(matrix[0])

    keep_cols = []

    # Step 1: compute column means with loops
    for j in range(cols):
        col_sum = 0
        for i in range(rows):
            col_sum += matrix[i][j]
        mean = col_sum / rows

        if mean > cutoff:
            keep_cols.append(j)

    # Step 2: build the filtered matrix (still using loops)
    filtered = []
    for i in range(rows):
        row_vals = []
        for j in keep_cols:
            row_vals.append(matrix[i][j])
        filtered.append(row_vals)

    return np.array(filtered)
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    col_means = np.mean(matrix, axis=0)        # 1 × num_columns
    mask = col_means > cutoff                  # Boolean mask
    return matrix[:, mask]                     # select only columns we keep
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    # Differences between consecutive days
    diffs = A[1:] - A[:-1]
    # Elementwise growth rates
    rates = diffs / A[:-1]
    # Round to 2 decimal places
    return np.round(rates, 2)
    ...


def with_leftover(A):
    # Money left over each day after buying as many whole shares as possible with $20
    # shares_per_day = floor(20 / price)
    # leftover_per_day = 20 - shares_per_day * price
    shares = np.floor(20 / A)
    leftovers = 20 - shares * A

    # Cumulative leftover over days
    cum_leftover = np.cumsum(leftovers)

    # Day where cumulative leftover is enough to buy 1 share at that day's price
    can_buy = cum_leftover >= A

    # Find first such day, if any
    indices = np.where(can_buy)[0]
    if indices.size == 0:
        return -1
    else:
        return int(indices[0])
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    # Try to infer column names in a flexible way
    player_col = [c for c in salary.columns if 'player' in c.lower() or 'name' in c.lower()][0]
    team_col   = [c for c in salary.columns if 'team'   in c.lower()][0]
    salary_col = [c for c in salary.columns if 'sal'    in c.lower()][0]

    # 1. Number of players
    num_players = len(salary)

    # 2. Number of teams
    num_teams = salary[team_col].nunique()

    # 3. Total salary
    total_salary = salary[salary_col].sum()

    # 4. Name of player with highest salary (assume no ties)
    idx_max = salary[salary_col].idxmax()
    highest_salary_player = salary.loc[idx_max, player_col]

    # 5. Average salary of the Los Angeles Lakers, rounded to 2 decimals
    lakers_mask = salary[team_col] == 'Los Angeles Lakers'
    avg_los = round(salary.loc[lakers_mask, salary_col].mean(), 2)

    # 6. Player with 5th lowest salary: "Name, Team"
    fifth_row = salary.sort_values(by=salary_col, ascending=True).iloc[4]
    fifth_lowest = f"{fifth_row[player_col]}, {fifth_row[team_col]}"

    # 7. Duplicated last names (ignoring suffixes like Jr., III, etc.)
    suffixes = {'Jr.', 'Jr', 'Sr.', 'Sr', 'II', 'III', 'IV', 'V'}

    def last_name_only(full_name):
        parts = full_name.split()
        if len(parts) == 0:
            return ''
        if parts[-1] in suffixes and len(parts) >= 2:
            return parts[-2]
        return parts[-1]

    last_names = salary[player_col].apply(last_name_only)
    duplicates = last_names.duplicated().any()

    # 8. Total salary of the team that has the highest paid player
    team_of_highest = salary.loc[idx_max, team_col]
    total_highest = salary.loc[salary[team_col] == team_of_highest, salary_col].sum()

    return pd.Series(
        {
            'num_players':  num_players,
            'num_teams':    num_teams,
            'total_salary': total_salary,
            'highest_salary': highest_salary_player,
            'avg_los':      avg_los,
            'fifth_lowest': fifth_lowest,
            'duplicates':   duplicates,
            'total_highest': total_highest,
        }
    )

    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    rows = []

    with open(fp) as fh:
        # Skip the header line
        header = fh.readline()

        for line in fh:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            # Remove all double quotes
            line = line.replace('"', '')

            # Split by commas
            parts = line.split(',')

            if len(parts) < 4:
                # Malformed beyond simple comma/quote issues; skip or handle as needed
                continue

            first = parts[0].strip()
            last = parts[1].strip()

            # Find weight and height as the first two non-empty numeric tokens after last name
            weight = None
            height = None
            geo_parts = []

            for token in parts[2:]:
                tok = token.strip()
                if tok == '':
                    # skip empty tokens caused by extra commas
                    continue

                if weight is None:
                    weight = float(tok)
                elif height is None:
                    height = float(tok)
                else:
                    geo_parts.append(tok)

            # Join whatever remains as geo (may contain commas originally)
            geo = ",".join(geo_parts)

            rows.append({
                "first": first,
                "last": last,
                "weight": float(weight),
                "height": float(height),
                "geo": geo,
            })

    return pd.DataFrame(rows, columns=["first", "last", "weight", "height", "geo"])

    ...
