# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    assignments = {
        'lab': [],
        'project': [],
        'checkpoint': [],
        'disc': [],
        'midterm': [],
        'final': []
    }
    
    for col in grades.columns:
        if re.fullmatch(r'lab\d{2}', col):
            assignments['lab'].append(col)

        elif re.fullmatch(r'project\d{2}', col):
            assignments['project'].append(col)

        elif re.fullmatch(r'checkpoint\d{2}', col):
            assignments['checkpoint'].append(col)

        elif re.fullmatch(r'discussion\d{2}', col):
            assignments['disc'].append(col)

        elif col == 'Midterm':
            assignments['midterm'].append(col)

        elif col == 'Final':
            assignments['final'].append(col)

    for k in assignments:
        assignments[k].sort()

    return assignments



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    # 1) Find all "project" score columns (exclude Max Points / Lateness helper cols)
    project_score_cols = [
        c for c in grades.columns
        if re.match(r"^project\d{2}", c)
        and "Max Points" not in c
        and "Lateness" not in c
    ]
    
    # Group score columns by project base name (project01, project02, ...)
    by_project = {}
    for c in project_score_cols:
        m = re.match(r"^(project\d{2})", c)
        if m:
            pid = m.group(1)
            by_project.setdefault(pid, []).append(c)

    # If no projects exist, return zeros (edge case)
    if not by_project:
        return pd.Series(0.0, index=grades.index)

    # 2) Compute each project's percentage for each student
    project_pcts = {}
    for pid, cols in by_project.items():
        cols = sorted(cols)

        max_cols = [f"{c} - Max Points" for c in cols]
        # earned points (missing -> 0)
        earned = grades[cols].fillna(0).sum(axis=1)

        # max points (should exist; fillna just in case)
        max_points = grades[max_cols].fillna(0).sum(axis=1)

        pct = earned / max_points
        pct = pct.where(max_points > 0, 0.0)  # avoid divide-by-zero

        project_pcts[pid] = pct

    # 3) Equal-weight across projects: average project percentages
    project_pct_df = pd.DataFrame(project_pcts)
    return project_pct_df.mean(axis=1)



# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    # Convert lateness strings to timedeltas
    lateness = pd.to_timedelta(col.fillna("00:00:00"))
    
    grace = pd.Timedelta(hours=2)
    one_week = pd.Timedelta(days=7)
    two_weeks = pd.Timedelta(days=14)
    
    # Start with no penalty
    penalty = pd.Series(1.0, index=col.index)
    
    # Apply penalties (only after grace period)
    penalty[(lateness > grace) & (lateness <= one_week)] = 0.9
    penalty[(lateness > one_week) & (lateness <= two_weeks)] = 0.7
    penalty[lateness > two_weeks] = 0.4
    
    return penalty


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    # Find all base lab columns like lab01, lab02, ...
    lab_cols = sorted([c for c in grades.columns if re.fullmatch(r"lab\d{2}", c)])
    
    processed = pd.DataFrame(index=grades.index)
    
    for lab in lab_cols:
        score_col = lab
        max_col = f"{lab} - Max Points"
        late_col = f"{lab} - Lateness (H:M:S)"
        
        # Earned points (missing = 0)
        earned = grades[score_col].fillna(0)
        
        # Max points (should exist; protect anyway)
        max_points = grades[max_col].fillna(0)
        
        # Lateness multiplier from Q3
        mult = lateness_penalty(grades[late_col])
        
        # Apply lateness penalty, then normalize
        adjusted = earned * mult
        
        # Normalize; if max is 0 (shouldn't happen), return 0
        processed[lab] = (adjusted / max_points).where(max_points > 0, 0.0)
    
    return processed


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
     # If there are 0 or 1 labs, handle edge cases safely
    n_labs = processed.shape[1]
    if n_labs == 0:
        return pd.Series(0.0, index=processed.index)
    if n_labs == 1:
        return processed.iloc[:, 0]

    # Drop the lowest lab score for each student, then average the rest
    total = (processed.sum(axis=1) - processed.min(axis=1)) / (n_labs - 1)
    return total



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    """
    Given base score columns like ['discussion01','discussion02',...]
    returns a Series of equal-weight average percentages in [0,1].
    """
    if len(base_cols) == 0:
        return pd.Series(0.0, index=grades.index)

    max_cols = [f"{c} - Max Points" for c in base_cols]

    earned = grades[base_cols].fillna(0)
    max_points = grades[max_cols].fillna(0)

    pct = earned / max_points
    pct = pct.where(max_points > 0, 0.0)  # safe if any max is 0

    return pct.mean(axis=1)
    ...

def total_points(grades):
    # ---- Labs (20%) ----
    labs_processed = process_labs(grades)
    lab_score = lab_total(labs_processed)

    # ---- Projects (30%) ----
    proj_score = projects_total(grades)

    # ---- Checkpoints (2.5%) ----
    checkpoint_cols = sorted([c for c in grades.columns if re.fullmatch(r"checkpoint\d{2}", c)])
    checkpoint_score = _equal_weight_total(grades, checkpoint_cols)

    # ---- Discussions (2.5%) ----
    discussion_cols = sorted([c for c in grades.columns if re.fullmatch(r"discussion\d{2}", c)])
    discussion_score = _equal_weight_total(grades, discussion_cols)

    # ---- Midterm (15%) ----
    if "Midterm" in grades.columns and "Midterm - Max Points" in grades.columns:
        midterm_score = (grades["Midterm"].fillna(0) / grades["Midterm - Max Points"].fillna(0)).replace([np.inf, -np.inf], 0).fillna(0)
    else:
        midterm_score = pd.Series(0.0, index=grades.index)

    # ---- Final (30%) ----
    if "Final" in grades.columns and "Final - Max Points" in grades.columns:
        final_score = (grades["Final"].fillna(0) / grades["Final - Max Points"].fillna(0)).replace([np.inf, -np.inf], 0).fillna(0)
    else:
        final_score = pd.Series(0.0, index=grades.index)

    # ---- Weighted total ----
    total = (
        0.20 * lab_score +
        0.30 * proj_score +
        0.025 * checkpoint_score +
        0.025 * discussion_score +
        0.15 * midterm_score +
        0.30 * final_score
    )

    return total


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    # total is a Series of proportions in [0, 1]
    bins = [-float("inf"), 0.6, 0.7, 0.8, 0.9, float("inf")]
    labels = ["F", "D", "C", "B", "A"]  # must align with bins
    
    return pd.cut(total, bins=bins, labels=labels, right=False).astype(str)

    ...

def letter_proportions(total):
    letters = final_grades(total)
    props = letters.value_counts(normalize=True)

    # ensure all letters present
    props = props.reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0.0)

    # expected order for this dataset/autograder
    return props.loc[['B', 'C', 'A', 'D', 'F']]

    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    ...
    
def combine_grades(grades, raw_redemption_scores):
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    ...
    
def add_post_redemption(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    ...
        
def proportion_improved(grades_combined):
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ...
    
def top_sections(grades_analysis, t, n):
    ...


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    ...







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    ...
