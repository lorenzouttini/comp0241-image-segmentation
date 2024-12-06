import numpy as np
# the goal of this routine is to return the minimum cost dynamic programming
# solution given a set of unary and pairwise costs
def dynamicProgram(unaryCosts, pairwiseCosts):

    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # define parent matrix - each element will contain the (vertical) index of
    # the node that preceded it on the path.  Since the first column has no
    # parents, we will leave it set to zeros.
    parents = np.zeros((nNodesPerPosition, nPosition), dtype=int)

    # FORWARD PASS

    # TODO:  fill in first column of minimum cost matrix
    for cNode in range(nNodesPerPosition):
        minimumCost[cNode, 0] = unaryCosts[cNode][0]
    # Note: No parents for the first column, so the `parents` matrix remains zero in the first column.


    # Now run through each position (column)
    for cPosition in range(1,nPosition):
        # run through each node (element of column)
        for cNode in range(nNodesPerPosition):
            # now we find the costs of all paths from the previous column to this node
            possPathCosts = np.zeros([nNodesPerPosition,1])
            for cPrevNode in range(nNodesPerPosition):
                # TODO  - fill in elements of possPathCosts
                possPathCosts[cPrevNode,0] = 0 
                # Compute path cost: transition cost + unary cost + previous minimum cost
                possPathCosts[cPrevNode, 0] = (
                    unaryCosts[cNode][cPosition] +              # Unary cost at the current node
                    minimumCost[cPrevNode, cPosition - 1] +     # Cost to the previous node
                    pairwiseCosts[cPrevNode][cNode]             # Transition cost between nodes
                )

            # TODO - find the minimum of the possible paths 
            minCost = np.min(possPathCosts)
            ind = np.argmin(possPathCosts)  # Index of the node that gives the minimum cost

            # Ensure there is only one minimum cost
            # assert len(np.where(possPathCosts == minCost)[0]) == 1, "Multiple minimum costs found!"

            # TODO - store the minimum cost in the minimumCost matrix
            minimumCost[cNode, cPosition] = minCost
            
            # TODO - store the parent index in the parents matrix
            parents[cNode, cPosition] = ind 

    #BACKWARD PASS

    # We will now fill in the bestPath vector
    bestPath = np.zeros([nPosition], dtype=int)

    # TODO Find the index of the overall minimum cost from the last column
    minCost = np.min(minimumCost[:, -1])  # Minimum cost in the last column
    minInd = np.argmin(minimumCost[:, -1])  # Index of the node with the minimum cost
    bestPath[-1] = minInd  # Last position in the best path corresponds to this index

    # TODO Initialize the parent of the last node
    bestParent = parents[minInd, -1]

    # TODO Run backwards through the cost matrix tracing the best path
    for cPosition in range(nPosition - 2, -1, -1):      # From second-to-last column to the first
        bestPath[cPosition] = bestParent                # Update the best path at the current position
        #bestParent = parents[bestParent, cPosition]     # Trace back to the next parent
        bestParent = int(parents[bestParent, cPosition])

    # Return the bestPath
    return bestPath


def dynamicProgramVec(unaryCosts, pairwiseCosts):
    
    # same preprocessing code
    
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    nNodesPerPosition = unaryCosts.shape[0]
    nPosition = unaryCosts.shape[1]

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])
    parents = np.zeros((nNodesPerPosition, nPosition), dtype=int)
    # Fill in the first column of the minimum cost matrix
    minimumCost[:, 0] = unaryCosts[:, 0]

    # TODO: fill this function in. (hint use tiling and perform calculations columnwise with matricies)
    # FORWARD PASS: Vectorized computation for each column
    for cPosition in range(1, nPosition):
        # Expand previous column costs into a matrix for pairwise cost addition
        prevCosts = minimumCost[:, cPosition - 1].reshape(-1, 1)  # Shape: (nNodesPerPosition, 1)
        totalCosts = prevCosts + pairwiseCosts  # Shape: (nNodesPerPosition, nNodesPerPosition)

        # Add unary costs for the current column (broadcasted)
        totalCosts += unaryCosts[:, cPosition].reshape(1, -1)

        # Find the minimum cost and parent for each node in the current column
        minimumCost[:, cPosition] = np.min(totalCosts, axis=0)
        parents[:, cPosition] = np.argmin(totalCosts, axis=0)

    # BACKWARD PASS: Reconstruct the best path
    bestPath = np.zeros(nPosition, dtype=int)
    bestPath[-1] = np.argmin(minimumCost[:, -1])  # Start with the minimum cost node in the last column

    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath




import numba
@numba.jit(nopython=True, parallel=True)
def compute_costs(minimumCost, unaryCosts, pairwiseCosts, nNodesPerPosition, nPosition, parents):
    # FORWARD PASS: Vectorized computation for each column
    for cPosition in range(1, nPosition):
        for currentNode in numba.prange(nNodesPerPosition):
            # Compute the range of costs for this node at this position
            prev_costs = minimumCost[:, cPosition - 1] + pairwiseCosts[:, currentNode]
            # Find the minimum cost and its parent
            min_cost = np.min(prev_costs)
            min_parent = np.argmin(prev_costs)
            # Update minimum cost and parent matrices
            minimumCost[currentNode, cPosition] = min_cost + unaryCosts[currentNode, cPosition]
            parents[currentNode, cPosition] = min_parent





def dynamicProgramVec_(unaryCosts, pairwiseCosts):
    
    # same preprocessing code
    
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    nNodesPerPosition = unaryCosts.shape[0]
    nPosition = unaryCosts.shape[1]

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])
    parents = np.zeros((nNodesPerPosition, nPosition), dtype=int)
    # Fill in the first column of the minimum cost matrix
    minimumCost[:, 0] = unaryCosts[:, 0]

    # TODO: fill this function in. (hint use tiling and perform calculations columnwise with matricies)
    # FORWARD PASS: Vectorized computation for each column
    for cPosition in range(1, nPosition):
        # Expand previous column costs into a matrix for pairwise cost addition
        prevCosts = minimumCost[:, cPosition - 1].reshape(-1, 1)  # Shape: (nNodesPerPosition, 1)
        totalCosts = prevCosts + pairwiseCosts  # Shape: (nNodesPerPosition, nNodesPerPosition)

        # Add unary costs for the current column (broadcasted)
        totalCosts += unaryCosts[:, cPosition].reshape(1, -1)

        # Find the minimum cost and parent for each node in the current column
        minimumCost[:, cPosition] = np.min(totalCosts, axis=0)
        parents[:, cPosition] = np.argmin(totalCosts, axis=0)
    
    for cPosition in range(1, nPosition):
        # Broadcast minimumCost to add pairwiseCosts
        prevCosts = minimumCost[:, cPosition - 1].reshape(-1, 1)  # Shape: (nNodesPerPosition, 1)
        totalCosts = prevCosts + pairwiseCosts  # Shape: (nNodesPerPosition, nNodesPerPosition)

        # Add unaryCosts and find minimum
        totalCosts += unaryCosts[:, cPosition].reshape(1, -1)
        minimumCost[:, cPosition] = np.min(totalCosts, axis=0)
        parents[:, cPosition] = np.argmin(totalCosts, axis=0)

    # BACKWARD PASS: Reconstruct the best path
    bestPath = np.zeros(nPosition, dtype=int)
    bestPath[-1] = np.argmin(minimumCost[:, -1])  # Start with the minimum cost node in the last column

    for cPosition in range(nPosition - 2, -1, -1):
        bestPath[cPosition] = parents[bestPath[cPosition + 1], cPosition + 1]

    return bestPath