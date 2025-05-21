# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Utils to get statistical information from a conversation

import math
from functools import reduce
import operator # For math.prod alternative if needed

from ..types import (
    Comment,
    CommentWithVoteInfo,
    VoteTally,
    GroupVoteTallies,
    VoteInfo,
    is_vote_tally_type, # Python equivalent of isVoteTallyType
)
from typing import List, Union # Union might not be strictly needed if VoteInfo handles it


def get_agree_rate(
    vote_tally: VoteTally,
    include_passes: bool,
    as_probability_estimate: bool = True,
) -> float:
    """
    Compute the probability of an agree vote for a given vote tally entry.
    """
    total_count = vote_tally.getTotalCount(include_passes)
    if total_count == 0 and not as_probability_estimate:
        return 0.0 # Avoid division by zero if not using estimate
    if as_probability_estimate:
        return (vote_tally.agreeCount + 1) / (total_count + 2)
    else:
        return vote_tally.agreeCount / total_count

def get_pass_rate(vote_tally: VoteTally, as_probability_estimate: bool = True) -> float:
    """
    Compute the probability of an pass vote for a given vote tally entry.
    """
    total_count = vote_tally.getTotalCount(True) # Always include passes for pass rate's total
    if total_count == 0 and not as_probability_estimate:
        return 0.0
    if as_probability_estimate:
        return ((vote_tally.passCount or 0) + 1) / (total_count + 2)
    else:
        return (vote_tally.passCount or 0) / total_count

def get_disagree_rate(
    vote_tally: VoteTally,
    include_passes: bool,
    as_probability_estimate: bool = True,
) -> float:
    """
    Compute the probability of a disagree vote for a given vote tally entry.
    """
    total_count = vote_tally.getTotalCount(include_passes)
    if total_count == 0 and not as_probability_estimate:
        return 0.0
    if as_probability_estimate:
        return (vote_tally.disagreeCount + 1) / (total_count + 2)
    else:
        return vote_tally.disagreeCount / total_count

def get_standard_deviation(numbers: List[float]) -> float:
    """
    Computes the standard deviation of a list of numbers.
    """
    if not numbers or len(numbers) <= 1:
        return 0.0  # Standard deviation of a single number or empty list is 0

    mean = sum(numbers) / len(numbers)
    # Use (n-1) for sample standard deviation, matching TypeScript
    variance = sum([(num - mean) ** 2 for num in numbers]) / (len(numbers) - 1)
    return math.sqrt(variance)

def _get_total_vote_count(group_vote_tallies: GroupVoteTallies, include_passes: bool) -> int:
    """
    Gets the total number of votes from GroupVoteTallies. Internal helper.
    """
    return sum(tally.getTotalCount(include_passes) for tally in group_vote_tallies.values())

def get_total_agree_rate(
    vote_info: VoteInfo,
    include_passes: bool,
    as_probability_estimate: bool = True,
) -> float:
    """
    Compute the probability of an agree vote for a given set of vote tallies.
    """
    if is_vote_tally_type(vote_info):
        # Type guard already narrows vote_info to VoteTally if true.
        # However, Python static checkers might need an explicit cast or isinstance check here
        # if is_vote_tally_type doesn't provide enough hints for them.
        # For runtime, it's fine.
        return get_agree_rate(vote_info, include_passes, as_probability_estimate)
    
    # If not VoteTally, it's GroupVoteTallies (vote_info: GroupVoteTallies)
    total_count = _get_total_vote_count(vote_info, include_passes)
    total_agree_count = sum(tally.agreeCount for tally in vote_info.values())
    
    if total_count == 0 and not as_probability_estimate:
        return 0.0
    if as_probability_estimate:
        return (total_agree_count + 1) / (total_count + 2)
    else:
        return total_agree_count / total_count

def get_total_pass_rate(
    vote_info: VoteInfo, as_probability_estimate: bool = True
) -> float:
    """
    Compute the probability of a pass vote for a given set of vote tallies.
    """
    if is_vote_tally_type(vote_info):
        return get_pass_rate(vote_info, as_probability_estimate)
    
    total_count = _get_total_vote_count(vote_info, True) # Always include passes for pass rate
    total_pass_count = sum((tally.passCount or 0) for tally in vote_info.values())

    if total_count == 0 and not as_probability_estimate:
        return 0.0
    if as_probability_estimate:
        return (total_pass_count + 1) / (total_count + 2)
    else:
        return total_pass_count / total_count

def get_total_disagree_rate(
    vote_info: VoteInfo,
    include_passes: bool,
    as_probability_estimate: bool = True,
) -> float:
    """
    Compute the probability of a disagree vote for a given set of vote tallies.
    """
    if is_vote_tally_type(vote_info):
        return get_disagree_rate(vote_info, include_passes, as_probability_estimate)

    total_count = _get_total_vote_count(vote_info, include_passes)
    total_disagree_count = sum(tally.disagreeCount for tally in vote_info.values())

    if total_count == 0 and not as_probability_estimate:
        return 0.0
    if as_probability_estimate:
        return (total_disagree_count + 1) / (total_count + 2)
    else:
        return total_disagree_count / total_count

def get_group_informed_consensus(comment: CommentWithVoteInfo) -> float:
    """
    Computes group informed (agree) consensus for a comment's vote tallies,
    computed as the product of the agree probabilities across groups.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo: # Also check if voteInfo is None
        raise TypeError("Group information is required for calculating group informed consensus.")
    
    # math.prod is available in Python 3.8+
    # Fallback for older Python: functools.reduce(operator.mul, iterable, 1)
    try:
        return math.prod(get_agree_rate(tally, True) for tally in comment.voteInfo.values())
    except AttributeError: # math.prod not found
        return reduce(operator.mul, (get_agree_rate(tally, True) for tally in comment.voteInfo.values()), 1.0)


def get_min_agree_prob(
    comment: CommentWithVoteInfo, as_probability_estimate: bool = True
) -> float:
    """
    Returns the minimum agree probability across groups.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo:
        raise TypeError("Group information is required for calculating minimum agree probability.")
    
    if not comment.voteInfo: # Should be caught by above, but good for type checker
        return 0.0 
        
    return min(
        get_agree_rate(tally, True, as_probability_estimate)
        for tally in comment.voteInfo.values()
    )

def get_group_informed_disagree_consensus(
    comment: CommentWithVoteInfo, as_probability_estimate: bool = True
) -> float:
    """
    Computes group informed (disagree) consensus for a comment's vote tallies
    computed as the product of disagree probabilities across groups.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo:
        raise TypeError("Group information is required for calculating group informed disagree consensus.")
    
    try:
        return math.prod(
            get_disagree_rate(tally, True, as_probability_estimate)
            for tally in comment.voteInfo.values()
        )
    except AttributeError: # math.prod not found
        return reduce(operator.mul, (
            get_disagree_rate(tally, True, as_probability_estimate)
            for tally in comment.voteInfo.values()
        ), 1.0)

def get_min_disagree_prob(
    comment: CommentWithVoteInfo, as_probability_estimate: bool = True
) -> float:
    """
    Returns the minimum disagree probability across groups.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo:
        raise TypeError("Group information is required for calculating the minimum disagree probability.")

    if not comment.voteInfo:
        return 0.0

    return min(
        get_disagree_rate(tally, True, as_probability_estimate)
        for tally in comment.voteInfo.values()
    )

def get_group_agree_prob_difference(
    comment: CommentWithVoteInfo, group: str, as_probability_estimate: bool = True
) -> float:
    """
    Computes the difference between the MAP probability estimate of agreeing within
    a given group as compared with the rest of the conversation.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo:
        raise TypeError("Group information is required for calculating group agreement probability difference.")
    
    if group not in comment.voteInfo:
        # Or handle as an error, depending on expected behavior
        # raise ValueError(f"Group '{group}' not found in comment's vote info.")
        return 0.0 # If group has no votes, its agree prob could be considered baseline

    group_agree_prob = get_agree_rate(
        comment.voteInfo[group], True, as_probability_estimate
    )

    other_groups_agree_count = 0
    other_groups_disagree_count = 0
    other_groups_pass_count = 0
    other_groups_total_votes_for_calc = 0 # for total_count in get_agree_rate denominator

    for g, tally in comment.voteInfo.items():
        if g != group:
            other_groups_agree_count += tally.agreeCount
            other_groups_disagree_count += tally.disagreeCount
            other_groups_pass_count += (tally.passCount or 0)
            # getTotalCount(include_passes=True) for probability estimate denominator
            other_groups_total_votes_for_calc += tally.getTotalCount(True) 


    if as_probability_estimate:
        other_groups_agree_prob = (other_groups_agree_count + 1) / (other_groups_total_votes_for_calc + 2)
    else:
        if other_groups_total_votes_for_calc == 0:
            other_groups_agree_prob = 0.0 # Or handle as per requirements for no "other" votes
        else:
            other_groups_agree_prob = other_groups_agree_count / other_groups_total_votes_for_calc
            
    return group_agree_prob - other_groups_agree_prob


def get_max_group_agree_prob_difference(comment: CommentWithVoteInfo) -> float:
    """
    Computes the maximal absolute value of `get_group_agree_prob_difference` across
    opinion groups present in comment.voteInfo.
    """
    if is_vote_tally_type(comment.voteInfo) or not comment.voteInfo:
        raise TypeError("Group information is required for calculating maximum group agreement probability difference.")

    if not comment.voteInfo: # No groups, no difference
        return 0.0

    group_names = comment.voteInfo.keys()
    if not group_names: # No groups, no difference
        return 0.0
        
    return max(
        abs(get_group_agree_prob_difference(comment, name)) for name in group_names
    )

def get_comment_vote_count(comment: Comment, include_passes: bool) -> int:
    """
    Computes the total vote count across opinion groups for a comment.
    """
    if not comment.voteInfo:
        return 0
    
    if is_vote_tally_type(comment.voteInfo):
        # voteInfo is VoteTally
        return comment.voteInfo.getTotalCount(include_passes)
    else:
        # voteInfo is GroupVoteTallies
        return _get_total_vote_count(comment.voteInfo, include_passes)
