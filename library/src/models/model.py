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

# Abstract class to interact with LLMs. Different implementations that call
# different LLM APIs will inherit this class and provide a concrete
# implementations that follow this structure. Then different models and
# model providers can be easily swapped in and out.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

# TSchema from "@sinclair/typebox" is a TypeScript-specific type for schema definition.
# In Python, this would typically be represented by a dictionary for a JSON schema,
# or a Pydantic model, or a similar schema validation library's construct.
# For the purpose of this translation, we'll use Dict or Any.
PySchema = Dict[str, Any] # Or simply use Any if schema structure is highly variable

# An abstract base class that defines how to interact with models.
class Model(ABC):
    # The best batch size to use for categorization.
    categorization_batch_size: int = 100

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """
        Abstract method for generating a text response based on the given prompt.
        Args:
            prompt: the instructions and data to process as a prompt
        Returns:
            the model response
        """
        pass

    @abstractmethod
    async def generate_data(self, prompt: str, schema: PySchema) -> Any:
        """
        Abstract method for generating structured data based on the given prompt.
        Args:
            prompt: the instructions and data to process as a prompt
            schema: the schema to use for the structured data (e.g., a JSON schema as a dict)
        Returns:
            the model response, structured according to the schema.
            The actual type would be `Static<typeof schema>` in TypeBox,
            here we use `Any` as it depends on the schema provided.
        """
        pass

# Specify which model will be called for different tasks. The tradeoff
# between speed and quality may be different for different modeling tasks.
@dataclass
class ModelSettings:
    default_model: Model
    summarization_model: Optional[Model] = None
    categorization_model: Optional[Model] = None
    grounding_model: Optional[Model] = None
