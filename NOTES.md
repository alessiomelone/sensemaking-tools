# Development Notes

## Feature Requests / Changes

### Opinion Weighting Without Vote Metadata
**Date:** 2025-08-18

**Issue:** Currently, when comments lack vote metadata (upvotes/downvotes), the model treats all comments equally during topic discovery and categorization. A single person's fringe opinion gets the same weight as a widely-held view.

**Proposed Change:** Modify the system to consider opinion prevalence even without explicit vote data by:
- Detecting similar/duplicate opinions in comment text
- Grouping semantically similar comments 
- Weighting topics/themes by frequency of similar sentiments
- Using text analysis to infer opinion distribution rather than relying solely on vote counts

**Impact:** This would make analysis more representative of actual opinion distribution in datasets without voting metadata.

**Files to modify:**
- `library/src/tasks/topic_modeling.ts` - Topic discovery weighting
- `library/src/tasks/categorization.ts` - Comment categorization logic
- `library/src/tasks/summarization.ts` - Summary generation to reflect opinion prevalence

### Topic-Driven Data Collection UX
**Date:** 2025-08-18

**Concept:** Advanced UX where users select specific topics and the system dynamically retrieves relevant datasets.

**User Flow:**
1. User selects specific topic (e.g., "Free public transport for students")
2. System searches relevant platforms based on selected topic:
   - Reddit discussions (r/changemyview, r/politics, etc.)
   - Social media posts and comments
   - Public consultation responses
   - Forum discussions
3. Dynamic dataset collection gathers all related posts and comments
4. Contextual analysis leverages user-selected topic for targeted sensemaking

**Implementation Requirements:**
- API integrations with data sources (Reddit API, Twitter API, etc.)
- Content filtering and relevance scoring algorithms
- Rate limiting and ethical data collection practices
- Dynamic topic modeling based on user-specified contexts
- Real-time analysis capabilities

**Benefits:**
- Real-time analysis of public opinion on specific issues
- No need for pre-collected CSV datasets
- More targeted and relevant sensemaking results

#### Cross-Platform Comparative Analysis
**Subsection:** Compare topics and argument support across different data sources and demographics.

**Concept:** Analyze how different communities/platforms approach the same topic with varying perspectives and priorities.

**Examples:**
- Reddit users vs Twitter users vs Parliament members on climate policy
- Different subreddit communities' takes on the same issue
- Academic discourse vs public opinion on scientific topics
- Regional differences in opinion distribution

**Implementation:**
- Parallel analysis of multiple datasets for the same topic
- Source-aware categorization and summarization
- Comparative visualization showing topic prevalence by source
- Demographic-specific argument mapping
- Cross-platform sentiment and reasoning analysis

### Interactive Opinion Exploration Agent
**Date:** 2025-08-20

**Concept:** Allow users to select a specific topic and engage in conversational exploration with an opinionated LLM agent that represents the viewpoints of people who support particular positions.

**Feature Description:**
Eventually, you could select a topic and 'talk' with the people that support some point, through an opinionated (feeding the comments again to the agent) LLM agent. In this way, you can delve deeper into the nuances of the argument.

**Implementation Approach:**
- Feed relevant comments/opinions to an LLM agent as context
- Configure the agent to represent the collective viewpoint of supporters
- Enable conversational interface for users to explore arguments in depth
- Allow switching between different opinion clusters/positions
- Maintain context of the original data while enabling dynamic exploration

**User Flow:**
1. User completes standard sensemaking analysis
2. System identifies distinct opinion clusters/positions on the topic
3. User selects a specific opinion cluster to explore
4. LLM agent is primed with comments from that cluster
5. User engages in conversation with the "representative" agent
6. Agent responds as if speaking for that group's perspective
7. User can switch to different opinion clusters for comparative exploration

**Benefits:**
- Deeper understanding of argument nuances and reasoning
- Interactive exploration beyond static summaries
- Ability to test counterarguments and edge cases
- More engaging way to explore complex opinion landscapes

### Social Media Comment Processing Pipeline
**Date:** 2025-08-20

**Feature:** Automated data pipeline to convert raw social media conversations into the standardized format required by the sensemaking system.

**Functionality:**
- Parse social media posts/threads (Twitter, Reddit, Facebook, etc.)
- Extract comment text, metadata (votes, timestamps, user info)
- Normalize data structure to match expected CSV format
- Handle nested comment threads and reply structures
- Output clean, analysis-ready dataset

**Input:** Raw social media conversation data
**Output:** Standardized CSV with columns: comment_text, votes, timestamp, user_id, parent_comment