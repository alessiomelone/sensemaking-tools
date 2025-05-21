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

import math
from dataclasses import dataclass, field
from typing import List, Callable, Dict, cast

from ..types import (
    Comment,
    CommentWithVoteInfo,
    GroupVoteTallies,
    is_group_vote_tallies_type,
    is_comment_with_vote_info_type, # Needed for filtering
)
from .stats_util import (
    get_group_agree_prob_difference,
    get_group_informed_consensus,
    get_group_informed_disagree_consensus,
    get_max_group_agree_prob_difference,
    get_min_agree_prob,
    get_min_disagree_prob,
    get_total_pass_rate,
)
# Assuming SummaryStats will be in summary_stats.py within the same directory
# If SummaryStats is not yet available, a placeholder would be needed.
# from .summary_stats import SummaryStats # This should be the actual import

# Placeholder for decimalToPercent from sensemaker_utils
def decimal_to_percent(decimal_value: float) -> str:
    if not 0 <= decimal_value <= 1:
        # Or raise ValueError, depending on desired strictness
        # print(f"Warning: decimal_value {decimal_value} is outside the typical 0-1 range for percentages.")
        pass
    return f"{decimal_value * 100:.0f}%"

# Placeholder for SummaryStats - replace with actual import when available
class SummaryStats:
    """
    Placeholder base class for GroupedSummaryStats.
    The actual SummaryStats class should be imported.
    """
    comments: List[Comment]
    # These attributes are assumed to be present based on their usage in GroupedSummaryStats
    filtered_comments: List[CommentWithVoteInfo] = field(default_factory=list)
    max_sample_size: int = 10 
    min_vote_count: int = 5 # Example default
    min_common_ground_prob: float = 0.65 # Example default
    min_agree_prob_difference: float = 0.15 # Example default
    min_uncertainty_prob: float = 0.1 # Example default


    def __init__(self, comments: List[Comment]):
        self.comments = comments
        # A basic way to populate filtered_comments for the placeholder
        # The actual SummaryStats would have its own logic, possibly more sophisticated.
        self.filtered_comments = []
        for c in comments:
            if is_comment_with_vote_info_type(c) and c.voteInfo is not None :
                 # Further filter based on min_vote_count, assuming get_comment_vote_count exists
                 # from .stats_util import get_comment_vote_count # circular import risk
                 # For placeholder, we might omit this or simplify
                 # if get_comment_vote_count(c, True) >= self.min_vote_count:
                self.filtered_comments.append(cast(CommentWithVoteInfo, c))


    @staticmethod
    def create(comments: List[Comment]) -> 'SummaryStats':
        return SummaryStats(comments)

    def top_k(
        self,
        sort_by: Callable[[CommentWithVoteInfo], float],
        k: int,
        filter_fn: Callable[[CommentWithVoteInfo], bool] = lambda c: True,
    ) -> List[CommentWithVoteInfo]:
        # Placeholder: actual implementation would use self.filtered_comments
        # and apply sorting and filtering.
        processed_comments = [c for c in self.filtered_comments if filter_fn(c)]
        processed_comments.sort(key=sort_by, reverse=True)
        return processed_comments[:k]


@dataclass
class GroupStats:
    name: str
    vote_count: int


class GroupedSummaryStats(SummaryStats):
    """
    This child class of the SummaryStats class provides the same abstract purpose
    (that is, serving as the interface to the RecursiveSummary abstraction),
    but is specifically tailored to group based summarization.
    """
    as_probability_estimate: bool = True

    def __init__(self, comments: List[Comment]):
        super().__init__(comments)
        # GroupedSummaryStats specific initialization can go here if needed
        # For example, ensuring all comments in filtered_comments have group vote tallies
        self.filtered_comments = [
            c for c in self.filtered_comments if is_group_vote_tallies_type(c.voteInfo)
        ]


    @staticmethod
    def create(comments: List[Comment]) -> 'GroupedSummaryStats':
        return GroupedSummaryStats(comments)

    def top_k(
        self,
        sort_by: Callable[[CommentWithVoteInfo], float],
        k: int = -1, # Default to using max_sample_size
        filter_fn: Callable[[CommentWithVoteInfo], bool] = lambda c: True,
    ) -> List[CommentWithVoteInfo]: # Return CommentWithVoteInfo instead of Comment
        if k == -1:
            k = self.max_sample_size
        
        # Ensure comments passed to sort_by and filter_fn are indeed CommentWithVoteInfo
        # and specifically those with GroupVoteTallies for this class.
        # self.filtered_comments should already contain CommentWithVoteInfo with GroupVoteTallies
        
        eligible_comments = [c for c in self.filtered_comments if filter_fn(c)]
        eligible_comments.sort(key=sort_by, reverse=True)
        return eligible_comments[:k]

    def get_common_ground_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents the common ground."""
        # Assumes comment.voteInfo is GroupVoteTallies due to class context / filtering
        return max(
            get_group_informed_disagree_consensus(comment, self.as_probability_estimate), # Pass as_probability_estimate
            self.get_common_ground_agree_score(comment)
        )

    def get_common_ground_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_common_ground_score(c),
            k=k,
            filter_fn=lambda c: (
                self.meets_common_ground_agree_threshold(c) or
                self.meets_common_ground_disagree_threshold(c)
            ),
        )

    def meets_common_ground_agree_threshold(self, comment: CommentWithVoteInfo) -> bool:
        return get_min_agree_prob(comment, self.as_probability_estimate) >= self.min_common_ground_prob

    def get_common_ground_agree_score(self, comment: CommentWithVoteInfo) -> float:
        return get_group_informed_consensus(comment) # Already uses as_probability_estimate by default from stats_util if not overridden

    def get_common_ground_agree_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_common_ground_agree_score(c),
            k=k,
            filter_fn=lambda c: self.meets_common_ground_agree_threshold(c),
        )

    def get_common_ground_no_comments_message(self) -> str:
        return (
            f"No statements met the thresholds necessary to be considered as a point of common "
            f"ground (at least {self.min_vote_count} votes, and at least "
            f"{decimal_to_percent(self.min_common_ground_prob)} agreement across groups)."
        )

    def get_common_ground_disagree_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: get_group_informed_disagree_consensus(c, self.as_probability_estimate),
            k=k,
            filter_fn=lambda c: self.meets_common_ground_disagree_threshold(c),
        )

    def meets_common_ground_disagree_threshold(self, comment: CommentWithVoteInfo) -> bool:
        return get_min_disagree_prob(comment, self.as_probability_estimate) >= self.min_common_ground_prob
    
    def get_group_representative_comments(self, group: str, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: get_group_agree_prob_difference(c, group, self.as_probability_estimate),
            k=k,
            filter_fn=lambda c: (
                get_min_agree_prob(c, self.as_probability_estimate) < self.min_common_ground_prob and
                get_group_agree_prob_difference(c, group, self.as_probability_estimate) > self.min_agree_prob_difference
            ),
        )

    def get_difference_of_opinion_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents a difference of opinions."""
        return get_max_group_agree_prob_difference(comment) # Already uses as_probability_estimate by default

    def get_difference_of_opinion_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_difference_of_opinion_score(c),
            k=k,
            filter_fn=lambda c: (
                get_min_agree_prob(c, self.as_probability_estimate) < self.min_common_ground_prob and
                # TS code had < self.minAgreeProbDifference, seems like a bug, should be >
                get_max_group_agree_prob_difference(c) > self.min_agree_prob_difference
            ),
        )

    def get_differences_of_opinion_no_comments_message(self) -> str:
        return (
            f"No statements met the thresholds necessary to be considered as a significant "
            f"difference of opinion (at least {self.min_vote_count} votes, and more than "
            f"{decimal_to_percent(self.min_agree_prob_difference)} difference in agreement rate between groups)."
        )

    def get_uncertain_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents an uncertain viewpoint based on pass votes."""
        # Ensure comment.voteInfo is not None before passing to get_total_pass_rate
        if comment.voteInfo is None:
             return 0.0 # Or raise error, or handle as per requirements
        return get_total_pass_rate(comment.voteInfo, self.as_probability_estimate)

    def get_uncertain_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_uncertain_score(c),
            k=k,
            filter_fn=lambda c: (
                c.voteInfo is not None and # Ensure voteInfo is not None
                get_total_pass_rate(c.voteInfo, self.as_probability_estimate) > self.min_uncertainty_prob
            ),
        )

    def get_stats_by_group(self) -> List[GroupStats]:
        group_name_to_stats: Dict[str, GroupStats] = {}
        # self.comments contains Comment objects. We need CommentWithVoteInfo with GroupVoteTallies.
        # This should ideally use self.filtered_comments which are already CommentWithVoteInfo
        # and filtered in __init__ to have GroupVoteTallies.
        for comment in self.filtered_comments: # Iterate over pre-filtered comments
            # voteInfo is already known to be GroupVoteTallies for comments in self.filtered_comments
            vote_info = cast(GroupVoteTallies, comment.voteInfo)
            for group_name, tally in vote_info.items():
                comment_vote_count = tally.getTotalCount(True) # includePasses = True
                if group_name in group_name_to_stats:
                    group_name_to_stats[group_name].vote_count += comment_vote_count
                else:
                    group_name_to_stats[group_name] = GroupStats(name=group_name, vote_count=comment_vote_count)
        return list(group_name_to_stats.values())
