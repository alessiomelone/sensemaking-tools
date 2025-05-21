# Copyright 2025 Google LLC
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

# Util class for models

import os

# The maximum number of times a task should be retried.
MAX_RETRIES: int = 3
# The maximum number of times an LLM call should be retried (it's higher to avoid rate limits).
MAX_LLM_RETRIES: int = 9
# How long in milliseconds to wait between API calls.
RETRY_DELAY_MS: int = 5000  # 5 seconds

# Set default vertex parallelism based on similarly named environment variable, or default to 2
parallelism_env_var: str | None = os.environ.get("DEFAULT_VERTEX_PARALLELISM")

default_vertex_parallelism_value: int = 2
if parallelism_env_var and parallelism_env_var.isdigit():
    default_vertex_parallelism_value = int(parallelism_env_var)
elif parallelism_env_var:
    print(f"Warning: Environment variable DEFAULT_VERTEX_PARALLELISM ('{parallelism_env_var}') is not a valid integer. Using default value {default_vertex_parallelism_value}.")

DEFAULT_VERTEX_PARALLELISM: int = default_vertex_parallelism_value
