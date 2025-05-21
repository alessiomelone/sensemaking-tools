from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union, Dict, Optional, Any, Literal

# TypeBox JSON Schema representation of a single topic record as a name, with no subtopics.
@dataclass
class FlatTopic:
    name: str

# TypeBox JSON Schema representation of a topic record as a name, with flat subtopics.
@dataclass
class NestedTopic:
    name: str
    subtopics: List[FlatTopic]

# Type representation of an abstract topic, either with or without subtopics.
Topic = Union[FlatTopic, NestedTopic]

# TypeBox JSON Schema representation of a comment id, together with a list of associated topics.
@dataclass
class TopicCategorizedComment:
    id: str
    topics: List[FlatTopic]

# TypeBox JSON Schema representation of a comment id, together with a list of associated topics and subtopics.
@dataclass
class SubtopicCategorizedComment:
    id: str
    topics: List[NestedTopic]

# Type representation of a comment id, together with a list of associated topics and possibly subtopics.
CommentRecord = Union[TopicCategorizedComment, SubtopicCategorizedComment]

# Describes the type of summarization to use.
class SummarizationType(Enum):
    GROUP_INFORMED_CONSENSUS = 0
    AGGREGATE_VOTE = 1

# Represents a portion of a summary, optionally linked to representative comments.
@dataclass
class SummaryContent:
    text: str
    type: Optional[str] = None
    title: Optional[str] = None
    citations: Optional[List[str]] = field(default_factory=list)
    subContents: Optional[List['SummaryContent']] = field(default_factory=list)


# Specifies the format for citations within a summary.
CitationFormat = Literal["XML", "MARKDOWN"]

# Aggregates a number of individual votes.
@dataclass
class VoteTally:
    agreeCount: int
    disagreeCount: int
    passCount: Optional[int] = None

    def getTotalCount(self, includePasses: bool) -> int:
        if includePasses:
            return self.agreeCount + self.disagreeCount + (self.passCount or 0)
        else:
            return self.agreeCount + self.disagreeCount

# A text that was voted on by different groups.
VoteInfo = Union[VoteTally, Dict[str, VoteTally]]

@dataclass
class Comment:
    id: str
    text: str
    voteInfo: Optional[VoteInfo] = None
    topics: Optional[List[Topic]] = None

@dataclass
class CommentWithVoteInfo(Comment):
    voteInfo: VoteInfo # Overrides the optional one in Comment


GroupVoteTallies = Dict[str, VoteTally]


class Summary:
    contents: List[SummaryContent]
    comments: List[Comment]

    def __init__(self, contents: List[SummaryContent], comments: List[Comment]):
        self.contents = contents
        self.comments = comments

    def getText(self, format: CitationFormat) -> str:
        # NOTE: The original TypeScript code uses `formatCitations` from "./tasks/utils/citation_utils"
        # and `filterSummaryContent` from "./sensemaker_utils".
        # These dependencies would need to be translated to Python and imported here.
        # For now, the MARKDOWN formatting will be simplified.
        return "\n".join([self.getContentText(content, format) for content in self.contents])

    def withoutContents(self, removeFn: callable[[SummaryContent], bool]) -> 'Summary':
        # NOTE: The original TypeScript code uses `filterSummaryContent` from "./sensemaker_utils".
        # This would need to be translated to Python.
        # For now, we'll implement a basic filter.
        
        # Helper function to recursively filter subContents
        def filter_content(content: SummaryContent) -> Optional[SummaryContent]:
            if removeFn(content):
                return None
            
            new_sub_contents = []
            if content.subContents:
                for sub_content in content.subContents:
                    filtered_sub = filter_content(sub_content)
                    if filtered_sub:
                        new_sub_contents.append(filtered_sub)
            
            # Create a new SummaryContent with potentially filtered subContents
            return SummaryContent(
                text=content.text,
                type=content.type,
                title=content.title,
                citations=content.citations,
                subContents=new_sub_contents if content.subContents else None
            )

        filtered_contents = []
        for sc in self.contents:
            processed_content = filter_content(sc)
            if processed_content:
                filtered_contents.append(processed_content)
        
        return Summary(filtered_contents, self.comments)

    def getContentText(self, content: SummaryContent, format: CitationFormat) -> str:
        result = ""
        if content.title:
            result += f"\n\n{content.title}\n"
        result += f"{content.text}{self.getCitationText(content, format)}"

        if content.subContents:
            for subcontent in content.subContents:
                result += self.getContentText(subcontent, format)
        
        return result

    def getCitationText(self, content: SummaryContent, format: CitationFormat) -> str:
        if not content.citations:
            return ""
        
        result = " "
        if format == "XML":
            for id_ in content.citations:
                result += f"<citation comment_id={id_}>"
        elif format == "MARKDOWN":
            # Simplified version, as formatCitations is not available.
            result += f"[{','.join(content.citations)}]"
        else:
            # In Python, it's more common to raise a ValueError for unsupported enum-like values.
            raise ValueError(f"Unsupported citation type: {format}")
        return result + " "


def is_vote_tally_type(data: Any) -> bool:
    return (
        isinstance(data, VoteTally) or
        (
            isinstance(data, dict) and
            "agreeCount" in data and isinstance(data["agreeCount"], int) and
            "disagreeCount" in data and isinstance(data["disagreeCount"], int) and
            (not "passCount" in data or data["passCount"] is None or isinstance(data["passCount"], int))
        )
    )

def is_group_vote_tallies_type(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return all(isinstance(key, str) and is_vote_tally_type(value) for key, value in data.items())

def is_comment_type(data: Any) -> bool:
    return (
        isinstance(data, Comment) or
        (
            isinstance(data, dict) and
            "id" in data and isinstance(data["id"], str) and
            "text" in data and isinstance(data["text"], str) and
            (not "voteInfo" in data or data["voteInfo"] is None or is_vote_tally_type(data["voteInfo"]) or is_group_vote_tallies_type(data["voteInfo"])) and
            (not "topics" in data or data["topics"] is None or (isinstance(data["topics"], list) and all(is_topic_type(t) for t in data["topics"])))
        )
    )

def is_comment_with_vote_info_type(data: Any) -> bool:
    return (
        is_comment_type(data) and
        hasattr(data, 'voteInfo') and data.voteInfo is not None and
        (is_vote_tally_type(data.voteInfo) or is_group_vote_tallies_type(data.voteInfo))
    ) or (
        isinstance(data, dict) and
        "id" in data and isinstance(data["id"], str) and # from is_comment_type
        "text" in data and isinstance(data["text"], str) and # from is_comment_type
        "voteInfo" in data and data["voteInfo"] is not None and
        (is_vote_tally_type(data["voteInfo"]) or is_group_vote_tallies_type(data["voteInfo"])) and
        (not "topics" in data or data["topics"] is None or (isinstance(data["topics"], list) and all(is_topic_type(t) for t in data["topics"])))
    )


def is_topic_type(data: Any) -> bool:
    # This is a simplified check. In a real scenario, you might want more robust validation,
    # potentially using a library if you were translating something like TypeBox directly.
    is_flat_topic = (
        isinstance(data, FlatTopic) or
        (isinstance(data, dict) and "name" in data and isinstance(data["name"], str) and "subtopics" not in data)
    )
    if is_flat_topic:
        return True

    is_nested_topic = (
        isinstance(data, NestedTopic) or
        (
            isinstance(data, dict) and "name" in data and isinstance(data["name"], str) and
            "subtopics" in data and isinstance(data["subtopics"], list) and
            all(is_flat_topic_dict(st) for st in data["subtopics"]) # Check subtopics are FlatTopic like
        )
    )
    return is_nested_topic

def is_flat_topic_dict(data: Any) -> bool:
    """Helper to check if a dict looks like a FlatTopic for is_topic_type"""
    return isinstance(data, dict) and "name" in data and isinstance(data["name"], str) and "subtopics" not in data


def is_comment_record_type(data: Any) -> bool:
    # This is a simplified check.
    # It checks if the data conforms to either TopicCategorizedComment or SubtopicCategorizedComment structure.
    if not (isinstance(data, dict) or isinstance(data, TopicCategorizedComment) or isinstance(data, SubtopicCategorizedComment)):
        return False

    id_present = hasattr(data, 'id') and isinstance(data.id, str) if not isinstance(data, dict) else 'id' in data and isinstance(data['id'], str)
    topics_present = hasattr(data, 'topics') and isinstance(data.topics, list) if not isinstance(data, dict) else 'topics' in data and isinstance(data['topics'], list)

    if not (id_present and topics_present):
        return False

    topics_list = data.topics if not isinstance(data, dict) else data['topics']

    if not topics_list: # topics list can be empty
        return True

    # Check if all topics are FlatTopic (for TopicCategorizedComment)
    # or NestedTopic (for SubtopicCategorizedComment)
    # This simplified check assumes homogeneity in the topics list for a given record.
    # A more accurate check would try to match one of the two specific types.
    is_topic_cat = all(is_topic_type(t) and not (hasattr(t, 'subtopics') or ('subtopics' in t and t['subtopics'])) for t in topics_list)
    if is_topic_cat: # if all are flat, it could be TopicCategorizedComment
        return True

    is_subtopic_cat = all(is_topic_type(t) and (hasattr(t, 'subtopics') or ('subtopics' in t and t['subtopics'])) for t in topics_list)
    if is_subtopic_cat: # if all are nested, it could be SubtopicCategorizedComment
        return True
    
    # If topics list contains a mix or doesn't strictly match, this simplified check might fail.
    # For a true TypeBox equivalent, one would need a more sophisticated schema validation.
    return False


# `checkDataSchema` from TypeScript relies on TypeBox for JSON schema validation.
# Python does not have a direct built-in equivalent for TypeBox.
# For robust schema validation in Python, one would typically use libraries like Pydantic,
# JSONSchema, or Marshmallow. Implementing a direct equivalent here is outside the scope
# of a simple translation. A placeholder function or a note is appropriate.

def check_data_schema(schema: Any, response: Any) -> bool:
    """
    Placeholder for TypeBox-like schema validation.
    In Python, you would typically use a library like Pydantic or jsonschema
    for this kind of validation.
    """
    # This is a very basic check, not equivalent to TypeBox.
    if schema == FlatTopic:
        return is_topic_type(response) and not (hasattr(response, 'subtopics') or ('subtopics' in response if isinstance(response, dict) else False))
    if schema == NestedTopic:
        return is_topic_type(response) and (hasattr(response, 'subtopics') or ('subtopics' in response if isinstance(response, dict) else False))
    if schema == CommentRecord:
        return is_comment_record_type(response)
    # Add more schema checks as needed for other types
    # print(f"Warning: Schema check for {schema} is not fully implemented.")
    return True # Defaulting to true for simplicity in this translation
