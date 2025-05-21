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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Set, Optional, TypeVar, Type, cast

from ..types import (
    Comment,
    CommentWithVoteInfo,
    is_comment_with_vote_info_type,
    Topic as TopicType, # Renamed to avoid conflict with TopicStats class name
    FlatTopic,
    NestedTopic,
)
from .stats_util import get_comment_vote_count, get_total_pass_rate

# Define a type variable for the class itself, used in the create method.
TStats = TypeVar('TStats', bound='SummaryStats')

# Placeholder for group_comments_by_subtopic from sensemaker_utils
def group_comments_by_subtopic(comments: List[Comment]) -> Dict[str, Dict[str, List[Comment]]]:
    """
    Groups comments by topic and subtopic.
    Placeholder implementation.
    """
    print("Warning: Using placeholder for group_comments_by_subtopic.")
    output: Dict[str, Dict[str, List[Comment]]] = {}
    for comment in comments:
        if comment.topics:
            for topic_obj in comment.topics:
                topic_name = topic_obj.name
                if topic_name not in output:
                    output[topic_name] = {}
                
                if isinstance(topic_obj, NestedTopic) and topic_obj.subtopics:
                    for subtopic_obj in topic_obj.subtopics:
                        subtopic_name = subtopic_obj.name
                        if subtopic_name not in output[topic_name]:
                            output[topic_name][subtopic_name] = []
                        output[topic_name][subtopic_name].append(comment)
                else: # FlatTopic or NestedTopic without subtopics treated as general for topic
                    if "_general_" not in output[topic_name]:
                         output[topic_name]["_general_"] = []
                    output[topic_name]["_general_"].append(comment)
        else: # No topics
            if "Uncategorized" not in output:
                output["Uncategorized"] = {"_general_": []}
            output["Uncategorized"]["_general_"].append(comment)
    return output


def get_75th_percentile(arr: List[float]) -> float:
    """
    Calculates the 75th percentile of a list of numbers.
    """
    if not arr:
        return 0.0 # Or raise error, depending on desired behavior for empty list
    
    sorted_arr = sorted(list(arr)) # Create a new sorted list
    index = (len(sorted_arr) - 1) * 0.75

    if math.floor(index) == index: # Integer index
        return sorted_arr[int(index)]
    
    # Interpolate if index is not an integer
    lower_index = math.floor(index)
    upper_index = lower_index + 1
    # Ensure upper_index is within bounds for small arrays
    if upper_index >= len(sorted_arr):
        return sorted_arr[lower_index] # Or handle as error / different logic
        
    return (sorted_arr[lower_index] + sorted_arr[upper_index]) / 2.0


@dataclass
class TopicStats:
    name: str
    comment_count: int
    # Using 'SummaryStats' as string because SummaryStats is defined later in this file (or imported)
    # This is a common way to handle forward references in type hints.
    summary_stats: 'SummaryStats' 
    subtopic_stats: Optional[List['TopicStats']] = field(default_factory=list)


class SummaryStats(ABC):
    """
    This class is the input interface for the RecursiveSummary abstraction, and
    therefore the vessel through which all data is ultimately communicated to
    the individual summarization routines.
    """
    comments: List[Comment]
    filtered_comments: List[CommentWithVoteInfo] # Comments with at least min_vote_count votes.
    
    min_common_ground_prob: float = 0.6
    min_agree_prob_difference: float = 0.3
    min_uncertainty_prob: float = 0.2 # Default, may be overridden in constructor
    as_probability_estimate: bool = False

    max_sample_size: int = 12
    min_vote_count: int = 20
    group_based_summarization: bool = True # Whether group data is used as part of the summary.

    def __init__(self, comments: List[Comment]):
        self.comments = comments
        
        # Filter comments that have vote info and meet min_vote_count
        candidate_filtered_comments: List[CommentWithVoteInfo] = []
        for comment in comments:
            if is_comment_with_vote_info_type(comment): # Check if it's CommentWithVoteInfo
                # Ensure voteInfo is not None before passing to get_comment_vote_count
                if comment.voteInfo is not None and \
                   get_comment_vote_count(comment, True) >= self.min_vote_count:
                    candidate_filtered_comments.append(cast(CommentWithVoteInfo, comment)) # Cast after check
            # else: comment is not CommentWithVoteInfo, so it's excluded.
        self.filtered_comments = candidate_filtered_comments
        
        # Adjust min_uncertainty_prob based on the 75th percentile of pass rates
        if self.filtered_comments: # Avoid error if no comments meet criteria
            pass_rates = [
                get_total_pass_rate(comment.voteInfo, self.as_probability_estimate)
                for comment in self.filtered_comments if comment.voteInfo is not None
            ]
            if pass_rates: # Ensure pass_rates is not empty before calculating percentile
                 top_quartile_pass_rate = get_75th_percentile(pass_rates)
                 self.min_uncertainty_prob = max(top_quartile_pass_rate, self.min_uncertainty_prob)
        # If no filtered_comments or no pass_rates, min_uncertainty_prob remains its default or pre-set value.


    @staticmethod
    def create(comments: List[Comment], cls: Type[TStats]) -> TStats: # Allow specifying class for creation
        """
        A static factory method that creates a new instance of SummaryStats
        or a subclass. Subclasses should override this.
        """
        # This base implementation should ideally be called by subclasses if they don't
        # fully override, or subclasses provide their own specific create.
        # The original TS version throws an error, implying subclasses MUST override.
        # For Python, we can make it more flexible or stick to that.
        # If we want to allow SummaryStats.create(comments, SpecificSubClass)
        if cls is SummaryStats:
             raise NotImplementedError("Cannot instantiate abstract class SummaryStats directly; use a subclass create method.")
        return cls(comments)


    @abstractmethod
    def get_common_ground_comments(self, k: Optional[int] = None) -> List[CommentWithVoteInfo]:
        pass

    @abstractmethod
    def get_common_ground_score(self, comment: CommentWithVoteInfo) -> float:
        pass

    @abstractmethod
    def get_common_ground_agree_comments(self, k: Optional[int] = None) -> List[CommentWithVoteInfo]:
        pass

    @abstractmethod
    def get_common_ground_no_comments_message(self) -> str:
        pass

    @abstractmethod
    def get_common_ground_disagree_comments(self, k: Optional[int] = None) -> List[CommentWithVoteInfo]:
        pass

    @abstractmethod
    def get_difference_of_opinion_comments(self, k: Optional[int] = None) -> List[CommentWithVoteInfo]:
        pass

    @abstractmethod
    def get_difference_of_opinion_score(self, comment: CommentWithVoteInfo) -> float:
        pass
    
    @abstractmethod
    def get_uncertain_comments(self, k: Optional[int] = None) -> List[CommentWithVoteInfo]:
        pass

    @abstractmethod
    def get_uncertain_score(self, comment: CommentWithVoteInfo) -> float:
        pass

    @abstractmethod
    def get_differences_of_opinion_no_comments_message(self) -> str:
        pass

    @property
    def vote_count(self) -> int:
        """The total number of votes across the entire set of input comments."""
        return sum(get_comment_vote_count(comment, True) for comment in self.comments if comment.voteInfo is not None)

    @property
    def comment_count(self) -> int:
        """The total number of comments in the set of input comments."""
        return len(self.comments)

    @property
    def contains_subtopics(self) -> bool:
        """Checks if any comment has nested topics (subtopics)."""
        for comment in self.comments:
            if comment.topics:
                for topic in comment.topics:
                    # Check if the topic is an instance of NestedTopic (which has 'subtopics' attribute)
                    # This relies on how Topic Union is structured and type-checked in types.py
                    if isinstance(topic, NestedTopic) and topic.subtopics:
                        return True
        return False

    def top_k(
        self,
        sort_by: Callable[[CommentWithVoteInfo], float], # Changed to CommentWithVoteInfo for consistency with subclasses
        k: Optional[int] = None,
        filter_fn: Callable[[CommentWithVoteInfo], bool] = lambda c: True,
    ) -> List[CommentWithVoteInfo]:
        """
        Returns the top k comments from filtered_comments according to the given metric.
        Note: TS version used this.comments. Changed to this.filtered_comments for consistency
        with how topK is usually used in subclasses. If base topK should operate on raw comments,
        the type of sort_by and filter_fn might need to be Comment, not CommentWithVoteInfo.
        Given subclasses operate on CommentWithVoteInfo, this seems more consistent.
        """
        if k is None:
            k = self.max_sample_size
        
        # Ensure comments are CommentWithVoteInfo and pass the filter_fn
        eligible_comments = [c for c in self.filtered_comments if filter_fn(c)]
        
        eligible_comments.sort(key=sort_by, reverse=True)
        return eligible_comments[:k]

    def _sort_topic_stats(
        self, topic_stats: List[TopicStats], sort_by_descending_count: bool = True
    ) -> List[TopicStats]:
        """
        Sorts topics and their subtopics based on comment count, with
        "Other" topics and subtopics going last in sortByDescendingCount order.
        """
        def sort_key(stat: TopicStats):
            # Primary key: 'Other' goes last if descending, first if ascending (effectively last due to reverse logic)
            # Secondary key: comment_count
            is_other = stat.name == "Other"
            count_val = stat.comment_count
            if sort_by_descending_count:
                return (is_other, -count_val) # False comes before True, so Other is last. Higher count is better (-count).
            else:
                return (is_other, count_val) # False comes before True. Lower count is better.

        topic_stats.sort(key=sort_key)

        for topic in topic_stats:
            if topic.subtopic_stats:
                topic.subtopic_stats.sort(key=sort_key)
        
        return topic_stats

    def get_stats_by_topic(self) -> List[TopicStats]:
        """
        Gets a sorted list of stats for each topic and subtopic.
        """
        # Use self.__class__ to refer to the actual class (e.g., GroupedSummaryStats) for the create call
        current_class = self.__class__

        comments_by_topic_subtopic = group_comments_by_subtopic(self.comments)
        topic_stats_list: List[TopicStats] = []

        for topic_name, subtopics_map in comments_by_topic_subtopic.items():
            all_topic_comments_set: Set[Comment] = set()
            subtopic_stats_list: List[TopicStats] = []

            for subtopic_name, comments_in_subtopic_list in subtopics_map.items():
                subtopic_comments_set = set(comments_in_subtopic_list)
                all_topic_comments_set.update(subtopic_comments_set)
                
                # Create SummaryStats instance for this specific subtopic's comments
                # Using current_class.create ensures that if this is called on a subclass instance,
                # the subclass's create method (and thus subclass instance) is created.
                subtopic_summary_stats = current_class.create(list(subtopic_comments_set), cls=current_class)

                subtopic_stats_list.append(
                    TopicStats(
                        name=subtopic_name,
                        comment_count=len(subtopic_comments_set),
                        summary_stats=subtopic_summary_stats,
                        # subtopic_stats is typically None/empty at the subtopic level itself
                    )
                )
            
            topic_summary_stats = current_class.create(list(all_topic_comments_set), cls=current_class)
            topic_stats_list.append(
                TopicStats(
                    name=topic_name,
                    comment_count=len(all_topic_comments_set),
                    subtopic_stats=subtopic_stats_list,
                    summary_stats=topic_summary_stats,
                )
            )
            
        return self._sort_topic_stats(topic_stats_list)
