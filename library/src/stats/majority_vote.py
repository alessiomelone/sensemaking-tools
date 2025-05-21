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
from dataclasses import field # Used in placeholder
from typing import List, Callable, cast # cast might be needed for voteInfo

from ..types import Comment, CommentWithVoteInfo, is_comment_with_vote_info_type
from .stats_util import get_total_agree_rate, get_total_disagree_rate, get_total_pass_rate
# Assuming SummaryStats will be in summary_stats.py within the same directory
# from .summary_stats import SummaryStats # This should be the actual import

# Placeholder for decimalToPercent from sensemaker_utils
def decimal_to_percent(decimal_value: float) -> str:
    if not 0 <= decimal_value <= 1:
        # print(f"Warning: decimal_value {decimal_value} is outside the typical 0-1 range for percentages.")
        pass # Allow values outside 0-1 for flexibility, though typically not expected for percentages
    return f"{decimal_value * 100:.0f}%"

# Placeholder for SummaryStats - replace with actual import when available
# This is the same placeholder used for group_informed.py
class SummaryStats:
    """
    Placeholder base class for MajoritySummaryStats.
    The actual SummaryStats class should be imported.
    """
    comments: List[Comment]
    filtered_comments: List[CommentWithVoteInfo] = field(default_factory=list)
    max_sample_size: int = 10
    min_vote_count: int = 5
    min_common_ground_prob: float = 0.65 # This will be overridden by MajoritySummaryStats
    min_agree_prob_difference: float = 0.15
    min_uncertainty_prob: float = 0.1 
    # group_based_summarization: bool = True # Placeholder, will be overridden
    # include_passes: bool = True # Placeholder, will be overridden
    # as_probability_estimate: bool = True # Placeholder, will be overridden


    def __init__(self, comments: List[Comment]):
        self.comments = comments
        self.filtered_comments = []
        for c_idx, c_val in enumerate(comments):
            # Ensure that the comment has voteInfo to be considered.
            # The actual SummaryStats would have more sophisticated filtering,
            # potentially using get_comment_vote_count from stats_util.
            if is_comment_with_vote_info_type(c_val) and c_val.voteInfo is not None:
                # For placeholder, assume min_vote_count check is done here or in a parent
                # from .stats_util import get_comment_vote_count # Avoid circular for placeholder
                # if get_comment_vote_count(c_val, True) >= self.min_vote_count:
                self.filtered_comments.append(cast(CommentWithVoteInfo, c_val))


    @staticmethod
    def create(comments: List[Comment]) -> 'SummaryStats':
        return SummaryStats(comments)

    def top_k(
        self,
        sort_by: Callable[[CommentWithVoteInfo], float],
        k: int,
        filter_fn: Callable[[CommentWithVoteInfo], bool] = lambda c: True,
    ) -> List[CommentWithVoteInfo]: # Return CommentWithVoteInfo
        processed_comments = [c for c in self.filtered_comments if filter_fn(c)]
        processed_comments.sort(key=sort_by, reverse=True)
        return processed_comments[:k]


class MajoritySummaryStats(SummaryStats):
    # Stats basis for the summary that is based on majority vote algorithms. Does not use groups.
    
    # Override defaults from SummaryStats placeholder or actual class
    min_common_ground_prob: float = 0.7
    # Agreement and Disagreement must be between these values to be difference of opinion.
    min_difference_prob: float = 0.4
    max_difference_prob: float = 0.6

    # Whether to include pass votes in agree and disagree rate calculations.
    include_passes: bool = False

    group_based_summarization: bool = False
    # This outlier protection isn't needed since we already filter out comments without many votes.
    as_probability_estimate: bool = False

    # Buffer between uncertainty comments and high/low alignment comments.
    uncertainty_buffer: float = 0.05
    
    # min_vote_count, max_sample_size, min_uncertainty_prob inherited or use placeholder defaults

    def __init__(self, comments: List[Comment]):
        super().__init__(comments)
        # MajoritySummaryStats specific initialization can go here if needed.
        # For example, ensuring filtered_comments are appropriate for non-group-based analysis.
        # The current placeholder for SummaryStats.__init__ already creates filtered_comments
        # with CommentWithVoteInfo. This class does not require GroupVoteTallies specifically.

    @staticmethod
    def create(comments: List[Comment]) -> 'MajoritySummaryStats':
        return MajoritySummaryStats(comments)

    def top_k(
        self,
        sort_by: Callable[[CommentWithVoteInfo], float],
        k: int = -1, # Default to using max_sample_size
        filter_fn: Callable[[CommentWithVoteInfo], bool] = lambda c: True,
    ) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        
        # self.filtered_comments should contain CommentWithVoteInfo objects
        eligible_comments = [c for c in self.filtered_comments if filter_fn(c) and c.voteInfo is not None]
        eligible_comments.sort(key=sort_by, reverse=True)
        return eligible_comments[:k]

    def get_common_ground_agree_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents when everyone agrees."""
        if comment.voteInfo is None: return 0.0
        return get_total_agree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate)

    def get_common_ground_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents the common ground."""
        if comment.voteInfo is None: return 0.0
        return max(
            self.get_common_ground_agree_score(comment),
            get_total_disagree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate)
        )

    def meets_common_ground_agree_threshold(self, comment: CommentWithVoteInfo) -> bool:
        if comment.voteInfo is None: return False
        return (
            get_total_agree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate) >=
                self.min_common_ground_prob and
            get_total_pass_rate(comment.voteInfo, self.as_probability_estimate) <=
                self.min_uncertainty_prob - self.uncertainty_buffer
        )

    def get_common_ground_agree_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_common_ground_agree_score(c),
            k=k,
            filter_fn=lambda c: self.meets_common_ground_agree_threshold(c)
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
            )
        )

    def get_common_ground_no_comments_message(self) -> str:
        return (
            f"No statements met the thresholds necessary to be considered as a point of common "
            f"ground (at least {self.min_vote_count} votes, and at least "
            f"{decimal_to_percent(self.min_common_ground_prob)} agreement)."
        )

    def get_uncertain_score(self, comment: CommentWithVoteInfo) -> float:
        """Returns a score indicating how well a comment represents an uncertain viewpoint based on pass votes."""
        if comment.voteInfo is None: return 0.0
        return get_total_pass_rate(comment.voteInfo, self.as_probability_estimate)

    def get_uncertain_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_uncertain_score(c),
            k=k,
            filter_fn=lambda c: (
                comment.voteInfo is not None and # Ensure voteInfo is not None
                get_total_pass_rate(comment.voteInfo, self.as_probability_estimate) >= self.min_uncertainty_prob
            )
        )

    def meets_common_ground_disagree_threshold(self, comment: CommentWithVoteInfo) -> bool:
        if comment.voteInfo is None: return False
        return (
            get_total_disagree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate) >=
                self.min_common_ground_prob and
            get_total_pass_rate(comment.voteInfo, self.as_probability_estimate) <=
                self.min_uncertainty_prob - self.uncertainty_buffer
        )

    def get_common_ground_disagree_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: get_total_disagree_rate(c.voteInfo, self.include_passes, self.as_probability_estimate) if c.voteInfo else 0.0,
            k=k,
            filter_fn=lambda c: self.meets_common_ground_disagree_threshold(c)
        )

    def get_difference_of_opinion_score(self, comment: CommentWithVoteInfo) -> float:
        if comment.voteInfo is None: return 0.0
        return (
            1.0 -
            abs(
                get_total_agree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate) -
                get_total_disagree_rate(comment.voteInfo, self.include_passes, self.as_probability_estimate)
            ) -
            get_total_pass_rate(comment.voteInfo, self.as_probability_estimate)
        )

    def get_difference_of_opinion_comments(self, k: int = -1) -> List[CommentWithVoteInfo]:
        if k == -1:
            k = self.max_sample_size
        return self.top_k(
            sort_by=lambda c: self.get_difference_of_opinion_score(c),
            k=k,
            filter_fn=lambda c: (
                c.voteInfo is not None and
                get_total_agree_rate(c.voteInfo, self.include_passes, self.as_probability_estimate) >=
                    self.min_difference_prob and
                get_total_agree_rate(c.voteInfo, self.include_passes, self.as_probability_estimate) <=
                    self.max_difference_prob and
                get_total_disagree_rate(c.voteInfo, self.include_passes, self.as_probability_estimate) >=
                    self.min_difference_prob and
                get_total_disagree_rate(c.voteInfo, self.include_passes, self.as_probability_estimate) <=
                    self.max_difference_prob and
                get_total_pass_rate(c.voteInfo, self.as_probability_estimate) <=
                    self.min_uncertainty_prob - self.uncertainty_buffer
            )
        )

    def get_differences_of_opinion_no_comments_message(self) -> str:
        min_threshold = decimal_to_percent(self.min_difference_prob)
        max_threshold = decimal_to_percent(self.max_difference_prob)
        return (
            f"No statements met the thresholds necessary to be considered as a significant "
            f"difference of opinion (at least {self.min_vote_count} votes, and both an agreement rate "
            f"and disagree rate between {min_threshold} and {max_threshold})."
        )
