import numpy as np

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

# P contains our training points.
# Each row represents one point with two coordinates: [x, y]
P = np.array([
    [1, 3],
    [2, 2],
    [3, 4],
    [4, 1],
    [5, 3],
    [2, 5],
    [6, 2]
], dtype=float)

# Class associated with each point in P
# P[0] belongs to class 0
# P[1] belongs to class 0
# P[2] belongs to class 1, etc.
labels = np.array([0, 0, 1, 1, 1, 0, 0])

# New point that we want to classify
q = np.array([3.0, 2.0])


# ---------------------------------------------------------
# EUCLIDEAN DISTANCE
# ---------------------------------------------------------

def euclidean(P, q):

    # P - q subtracts q from every point in P.
    #
    # Example:
    # [1, 3] - [3, 2] = [-2, 1]
    #
    # (P - q) ** 2 squares every difference.
    #
    # [-2, 1] -> [4, 1]
    #
    # sum(axis=1) adds the values of each ROW.
    #
    # [4, 1] -> 5
    #
    # np.sqrt() gives the Euclidean distance:
    #
    # sqrt(5) = 2.236

    return np.sqrt(((P - q) ** 2).sum(axis=1))


# ---------------------------------------------------------
# MANHATTAN DISTANCE
# ---------------------------------------------------------

def manhattan(P, q):

    # P - q calculates the coordinate differences.
    #
    # np.abs() makes all differences positive.
    #
    # Example:
    # [1, 3] - [3, 2] = [-2, 1]
    # abs([-2, 1]) = [2, 1]
    #
    # sum(axis=1) adds the differences for each point.
    #
    # [2, 1] -> 3

    return np.abs(P - q).sum(axis=1)


# ---------------------------------------------------------
# K-NEAREST NEIGHBOURS
# ---------------------------------------------------------

def knn_vote(P, labels, q, k, dist, weighted=False):

    # Calculate the distance between q and every point in P.
    #
    # If dist = euclidean, Euclidean distance is used.
    # If dist = manhattan, Manhattan distance is used.
    d = dist(P, q)


    # np.argsort(d) returns the INDICES that would sort d
    # from smallest distance to largest distance.
    #
    # Example:
    #
    # d = [2.23, 1.0, 2.0]
    #
    # np.argsort(d)
    # -> [1, 2, 0]
    #
    # because:
    # d[1] = 1.0
    # d[2] = 2.0
    # d[0] = 2.23
    #
    # [:k] keeps only the first k indices.
    #
    # These correspond to the k nearest neighbours.

    idx = np.argsort(d, kind="stable")[:k]


    # Decide how much each neighbour contributes to the vote.
    #
    # Normal KNN:
    # Every neighbour gets weight 1.
    #
    # Weighted KNN:
    # A closer neighbour gets a larger weight:
    #
    # weight = 1 / distance

    if weighted:
        w = 1 / d[idx]
    else:
        w = np.ones(k)


    # labels[idx] gives us the classes of the k nearest neighbours.
    #
    # Example:
    #
    # idx = [1, 3, 2]
    # labels[idx] = [0, 1, 1]
    #
    # np.bincount counts the votes for each class.
    #
    # Without weights:
    #
    # class 0 -> 1 vote
    # class 1 -> 2 votes
    #
    # With weights, it adds the weights instead of simply
    # counting the number of neighbours.

    totals = np.bincount(
        labels[idx],
        weights=w,
        minlength=labels.max() + 1
    )


    # totals might look like:
    #
    # [1, 2]
    #
    # meaning:
    # class 0 has score 1
    # class 1 has score 2
    #
    # argmax() returns the POSITION of the largest value.
    #
    # np.argmax([1, 2]) -> 1
    #
    # Therefore the predicted class is class 1.

    return totals.argmax()


# ---------------------------------------------------------
# CALCULATE EUCLIDEAN DISTANCES
# ---------------------------------------------------------

d = euclidean(P, q)


# ---------------------------------------------------------
# DISPLAY THE POINTS FROM NEAREST TO FARTHEST
# ---------------------------------------------------------

# argsort gives us the indices ordered according to distance.
for i in np.argsort(d, kind="stable"):

    print(
        f"P{i+1}  "
        f"class {labels[i]}   "
        f"d^2 = {d[i]**2:4.0f}   "
        f"d = {d[i]:.3f}   "
        f"d1 = {manhattan(P, q)[i]:.0f}"
    )


# ---------------------------------------------------------
# TEST DIFFERENT VALUES OF K
# ---------------------------------------------------------

# Test KNN with k = 1, 3, 5 and 7.
#
# Here we use Euclidean distance and normal voting.

for k in (1, 3, 5, 7):

    prediction = knn_vote(
        P,
        labels,
        q,
        k,
        euclidean
    )

    print(f"k={k}: class {prediction}")


# ---------------------------------------------------------
# KNN WITH MANHATTAN DISTANCE
# ---------------------------------------------------------

# Same KNN algorithm, but this time distance is calculated
# using Manhattan distance instead of Euclidean distance.

print(
    "Manhattan, k=3:",
    knn_vote(P, labels, q, 3, manhattan)
)


# ---------------------------------------------------------
# WEIGHTED KNN
# ---------------------------------------------------------

# Here k = 5, but closer neighbours have more influence.
#
# weight = 1 / distance

print(
    "weighted, k=5:",
    knn_vote(P, labels, q, 5, euclidean, weighted=True)
)


# ---------------------------------------------------------
# SHOW THE WEIGHTED SCORES
# ---------------------------------------------------------

# Find the 5 closest points.
idx5 = np.argsort(d, kind="stable")[:5]

# labels[idx5] gives their classes.
#
# 1 / d[idx5] gives the weight of each neighbour.
#
# bincount then adds these weights separately for
# class 0 and class 1.

weighted_totals = np.bincount(
    labels[idx5],
    weights=1 / d[idx5]
)

print(
    "weighted totals (class 0, class 1):",
    weighted_totals
)
