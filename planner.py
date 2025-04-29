import cvxpy as cp
import numpy as np
from collections import defaultdict

from intervals import all_intervals, escape_activities_for_planning


def planning(target_total_duration = 20000, target_total_duration_error_margin = 0.2, minimal_portion = None, tracks = [], maximal_portion = None, init_vals = dict()):
    """
    target_total_duration: the planned total duration (in seconds)
    target_total_duration_error_margin: how much difference from target_total_duration is tolerated. 0.2 means 80%-120% will be accepted
    minimal_portion: the minimal portion per track in the total number of tracks
    """
    # Result: a dictionary mapping each key to a dict of activity durations
    activity_totals = {}

    for track_name in tracks:
        intervals = all_intervals[track_name]
        track_activity_durations = defaultdict(float)
        for activity, start, end in intervals:
            if activity not in escape_activities_for_planning:
                track_activity_durations[activity] += end - start
        activity_totals[track_name] = dict(track_activity_durations)

    # Get sorted list of all unique activities
    all_activities = sorted({act for track in activity_totals.values() for act in track})

    all_track_names = sorted(activity_totals.keys())

    # default minimal_portion
    if minimal_portion == None:
        minimal_portion = 1.0/(len(all_track_names)*2)

    if maximal_portion == None:
        maximal_portion = 2*1.0/(len(all_track_names))

    print(all_activities)

    # Map each track into a vector of durations in activity order
    time_matrix = np.array([
        [activity_totals[track_name].get(activity, 0.0) for activity in all_activities]
        for track_name in all_track_names
    ])

    print(time_matrix)


    non_zero_columns = np.count_nonzero(np.any(time_matrix != 0, axis=0))


    x = cp.Variable(time_matrix.shape[0], integer=True)  # servings of A, B, C

    ##mean = cp.sum(totals) / non_zero_columns
    #objective = cp.Minimize(cp.sum_squares(totals - mean)) 

    offsets = np.zeros(len(all_activities))

    for i, activity in enumerate(all_activities):
        if activity in init_vals:
            offsets[i]+=init_vals[activity]

    totals = x @ time_matrix + offsets

    total_duration = cp.sum(totals)
    mean = total_duration / non_zero_columns

    time_deviation = cp.square(total_duration - target_total_duration)
    imbalance = cp.sum_squares(totals - mean)

    #alpha = 0  # or 0.2, 0.8 depending on your priority
    #objective = cp.Minimize(alpha * time_deviation + (1 - alpha) * imbalance)
    objective  = cp.Minimize(imbalance) 
    #objective = cp.Minimize(cp.sum_squares(totals - mean)) 

    total_copies = cp.sum(x)

    if len(init_vals)>0:
        portion_constraints = []
    else:
        portion_constraints = [
            x >= minimal_portion * total_copies,
            x <= maximal_portion * total_copies
        ]

    constraints = [
        cp.sum(totals) >= (1 - target_total_duration_error_margin) * target_total_duration,
        cp.sum(totals) <= (1 + target_total_duration_error_margin) * target_total_duration,
        x >= 1,
    ]
    constraints += portion_constraints

    # Solve using a MIP solver like ECOS_BB
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB)

    # Output
    print("\nProblem status:", prob.status)
    if prob.status in ["optimal", "optimal_inaccurate"]:
        for i, val in enumerate(x):
            print(f"{all_track_names[i]} = {val.value:.0f}")

        Y = x @ time_matrix
        print("\n")
        for i, val in enumerate(Y):
            print(f"{all_activities[i]} total = {val.value:.2f}")

        print(f"Total time = {cp.sum(Y).value + np.sum(offsets):.2f}")
    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    #planning(target_total_duration_error_margin = 0.1, tracks = list(all_intervals.keys()), minimal_portion = 0.18, maximal_portion = 0.3, init_vals = {"walk":1000})
    planning(target_total_duration_error_margin = 0.1, tracks = list(all_intervals.keys()), minimal_portion = 0.18, maximal_portion = 0.3)