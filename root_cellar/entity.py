import re
from abc import ABC
from typing import List, Optional, Dict, Any, Literal, Union, AnyStr, ClassVar
from pydantic import BaseModel,Field,SerializeAsAny,root_validator

from .llm import LLMType

class SimpleEntity(BaseModel):
    """
    A minimal entity definition for recording a single entity mentioned in a chat thread.
    """

    name: str = Field(
        default=...,
        description="The name of this entity."
    )
    description: str = Field(
        default=...,
        description="A description of this entity. Write more detailed descriptions for more important entities."
    )

class SimpleEntityList(BaseModel):
    """A single list of entities."""

    entities: List[SimpleEntity] = Field(
        description="A list of all the entities."
    )

class Entity:
    """A single entity mentioned in a chat thread."""

    id: str
    """Unique identifier for this entity."""
    name: str
    """The name of this entity."""
    description: str
    """A description of this entity. Use more detail for more important entities."""
    last_used_in_summary: int
    """
    The index of the most recent summary where this entity appeared, whether or not
    the entity is directly mentioned in the summary text.
    """
    last_injected_with_message: int # maybe not put this here?
    """
    The index of the last AI message where this entity was injected as context.
    If the entity is injected at the beginning via a summary, this value will be -1.
    """
    is_in_context: bool
    """Whether or not this entity is currently injected into the chat thread."""

class GenEntity(BaseModel):
    """
    A simplified entity type containing only the parts of entities that we want to
    have the LLM generate. We keep other programmatic bookkeeping stuff out of here
    to avoid unnecesary cognitive load on the LLM.
    """

    name: str = Field(
        default=...,
        description="The name of this entity."
    )
    description: str = Field(
        default=...,
        description="A description of this entity. Write more detailed descriptions for more important entities."
    )

class GenEntityList(BaseModel):
    """
    A simplified entity list type containing only the parts of entities that we want to
    have the LLM generate. We keep other programmatic bookkeeping stuff out of here
    to avoid unnecesary cognitive load on the LLM.
    """

    entities: List[GenEntity] = Field(
        default=[],
        description="A list of all the entities."
    )

class EntityManager(BaseModel, ABC):
    """
    Keeps track of entities that have been extracted from a chat thread and manages
    injecting them into context as needed.
    """
    # type name for deserialization
    entity_manager_class:Literal['base'] = "base"

    def update_entities(self, messages: List[Dict[str, str]], prior_summaries: List[Dict[str, str]] = []) -> Any:
        """
        Update a list of previously-mentioned entities, optionally including a list of
        older summaries as context.

        Args:
            messages: a list of messages to extract/update entities from
            prior_summaries: a list of older summaries to be used as context when updating

        Returns:
            the updated entity list
        """
        raise NotImplementedError("Method not implemented!")

class SimpleEntityManager(EntityManager):
    """
    Keeps track of entities that have been extracted from a chat thread as a single monolithic
    block of text.
    """
    # type name for deserialization
    entity_manager_class:Literal['SimpleEntityManager'] = "SimpleEntityManager"

    llm: SerializeAsAny[LLMType] = Field(
        default=...,
        discriminator='llm_class',
        description="LLM instance used to generate the entity list")
    entity_list: Optional[str] = Field(None, description="Current entity list as a free-form list in a single string")
    prompt_entity_list: str = Field(
        default=(
            "You are creating a list of all important entities mentioned thus far "
            "and a brief description of each. You will be given any relevant prior context and the user "
            "will provide the messages from which you should extract or update entities. For people, "
            "include a brief description of their personalities and appearance. Write more detailed "
            "descriptions for more important entities.\n\n"
            "Prior context:\n"
            "{context}\n\n"
            "Existing list of entities to be updated:\n"
            "{entities}\n\n"
            "Now the user will provide you with the messages from which you should extract entity information. "
            "Respond only with a list of significant entities and a brief description of each entity, no "
            "additional commentary."
        ),
        description="System prompt used when asking the LLM to produce or update the entity list"
    )

    @root_validator(pre=True)
    def ensure_llm_present(cls, values):
        if 'llm' not in values or values.get('llm') is None:
            raise ValueError("llm is required for SimpleEntityManager")
        return values

    def update_entities(self, messages: List[Dict[str, str]], prior_summaries: List[Dict[str, str]] = []) -> str:
        """
        Update a free-form list of previously-mentioned entities, optionally including a list of
        older summaries as context.

        Args:
            messages: a list of messages to extract/update entities from
            prior_summaries: a list of older summaries to be used as context when updating

        Returns:
            the updated entity list (string)
        """
        # if no prior context, just put 'No prior context.' in as a placeholder
        if not prior_summaries:
            prior_summaries = [{'content': "No prior context."}]

        # if no existing entity list, put in a placeholder
        old_ent_list = self.entity_list
        if old_ent_list is None or len(old_ent_list.strip()) == 0:
            old_ent_list = "No prior entity list available."

        # construct system prompt
        sys_prompt = {
            'role': 'system',
            'content': self.prompt_entity_list.format(
                context="\n\n".join([ps['content'] for ps in prior_summaries]),
                entities=old_ent_list
            )
        }
        user_prompt = {
            'role': 'user',
            'content': "Please update the entity list using information from the following messages:\n\n"
                       + "\n\n".join([m['content'] for m in messages])
        }

        # generate the entity list using the provided LLM interface
        llm_response = self.llm.generate_instruct(
            messages=[sys_prompt, user_prompt],
            respond=True,
            response_role="assistant",
            stream=False
        )

        # pull the first/only result off the generator and strip whitespace
        self.entity_list = next(llm_response)['response'].strip()
        return self.entity_list

class JSONEntityManager(EntityManager):
    """
    Keeps track of entities that have been extracted from a chat thread as a single monolithic
    block of text.
    """
    # type name for deserialization
    entity_manager_class:Literal['JSONEntityManager'] = "JSONEntityManager"
    
    # pulls role names out of a string representation of a thread
    list_pattern: ClassVar[re.Pattern[AnyStr]] = re.compile(r"\[([^\]]*)")

    llm: SerializeAsAny[LLMType] = Field(
        default=...,
        discriminator='llm_class',
        description="LLM instance used to generate the entity list")
    entity_list: GenEntityList = Field(
        default=GenEntityList(entities=[]), 
        description="A list of the entities mentioned in this chat thread."
    )
    prompt_entity_list: str = Field(
        default=(
            "You are creating a list of all important entities mentioned thus far "
            "and a brief description of each. You will be given any relevant prior context. "
            "You will also have a copy of the current entity list. Finally, you "
            "will receive the messages from which you should extract or update entities. For people, "
            "include a brief description of their personalities and appearance. Write more detailed "
            "descriptions for more important entities.\n\n"
            "Prior context:\n"
            "{context}\n\n"
            "Existing list of entities to be updated:\n"
            "{entities}\n\n"
            "Messages from which you should extract or update entity information:\n"
            "{messages}"
        ),
        description="System prompt used when asking the LLM to produce or update the entity list"
    )

    @root_validator(pre=True)
    def ensure_llm_present(cls, values):
        if 'llm' not in values or values.get('llm') is None:
            raise ValueError("llm is required for SimpleEntityManager")
        return values

    def update_entities(self, messages: List[Dict[str, str]], prior_summaries: List[Dict[str, str]] = []) -> GenEntityList:
        """
        Update a list of previously-mentioned entities, optionally including a list of
        older summaries as context.

        Args:
            messages: a list of messages to extract/update entities from
            prior_summaries: a list of older summaries to be used as context when updating

        Returns:
            the updated entity list
        """
        # if no prior context, just put 'No prior context.' in as a placeholder
        if not prior_summaries:
            prior_summaries = [{'content': "No prior context."}]

        # if no existing entity list, put in a placeholder
        old_ent_list = self.entity_list.entities
        
        if old_ent_list is None or len(old_ent_list) == 0:
            ent_txt = "No prior entity list available."
        else:
            ent_txt = "\n\n".join([ent.name + ": " + ent.description for ent in self.entity_list.entities])

        # construct system prompt
        sys_prompt = {
            'role': 'system',
            'content': self.prompt_entity_list.format(
                context="\n\n".join([ps['content'] for ps in prior_summaries]),
                entities=ent_txt,
                messages="\n\n".join([m['content'] for m in messages])
            )
        }
        # make blank list to hold updated entities
        updated_list = GenEntityList(entities=[])
        # for each entity
        for entity in self.entity_list.entities:
            user_prompt = {
                'role': 'user',
                'content': f"Should entity '{entity.name}' be updated based on the messages? Respond ONLY with ##YES## or ##NO##."
            }
            update_decision = self.llm.generate_instruct(
                messages=[sys_prompt, user_prompt],
                response_role="assistant",
                stream=False
            )
            print("Checking entity: " + entity.name)
            llm_response = ""
            for chunk in update_decision:
                llm_response += chunk['response']
            print("Needs updating? " + llm_response)

            # if model decided to edit the entity
            if "##YES##" in llm_response:
                response_msg = {
                    'role':'assistant',
                    'content': llm_response
                }
                update_prompt = {
                    'role': 'user',
                    'content': f"Please update entity '{entity.name}'. Respond ONLY with the updated entity name and description formatted as 'name: description'."
                }
                response_start = {
                    'role': 'assistant',
                    'content': f"{entity.name}: "
                }
                updated_entity = self.llm.generate_instruct(
                    messages=[sys_prompt, user_prompt, response_msg, update_prompt, response_start],
                    respond=False,
                    stream=False
                )
                
                llm_response = ""
                for chunk in updated_entity:
                    llm_response += chunk['response']
                # parse response
                updated_entity = GenEntity(
                    name=entity.name,
                    description=llm_response.strip()
                )
                print("Updated entity:")
                print(llm_response)
                # add to updated list
                updated_list.entities.append(updated_entity)
            else:
                # not updating, so just put the old entity in the list
                updated_list.entities.append(entity)
        # now check if there are any new entities we need to extract
        new_ent_prompt = {
            'role': 'user',
            'content': (
                "Based on the messages and the existing entity list, are there any new entities "
                "we need to add to the entity list? Respond ONLY with a comma-separated list containing "
                "any new entities that should be added. Surround the list with square brackets, like so:\n\n"
                "[name1, name2, name3, name4]\n\n"
                "If there are no new entities to add, respond ONLY with an empty list, like so:\n\n"
                "[]"
            )
        }
        new_entity_response = self.llm.generate_instruct(
            messages=[sys_prompt, new_ent_prompt],
            response_role="assistant",
            stream=False
        )
        llm_response = ""
        for chunk in new_entity_response:
            llm_response += chunk['response']
        print("List of entities to add:")
        print(llm_response)
        # parse the list of entities to add, ignoring any non-list stuff
        list_txt = self.list_pattern.search(llm_response)
        if not list_txt:
            print("Warning: LLM may have made a mistake writing out a list of entities to add!")
            print("LLM response was: " + llm_response)
            return updated_list
        # get just the list
        list_txt = list_txt.group(1).strip()
        # if list is empty
        if len(list_txt) == 0:
            return updated_list
        # split on commas to get entities
        list_names = list_txt.split(",")
        # strip whitespace in case LLM put some in
        list_names = [name.strip() for name in list_names]
        # get unique names in case LLM repeated any
        list_names = set(list_names)
        # make sure these names are actually not in the existing list
        list_names = list_names.difference([e.name for e in updated_list.entities])
        
        # if nothing new to add
        if len(list_names) == 0:
            return updated_list
        
        # generate descriptions for new entities
        for name in list_names:
            update_prompt = {
                'role': 'user',
                'content': f"Please update entity '{name}'. Respond ONLY with the updated entity name and description formatted as 'name: description'."
            }
            response_start = {
                'role': 'assistant',
                'content': f"{name}: "
            }
            updated_entity = self.llm.generate_instruct(
                messages=[sys_prompt, update_prompt, response_start],
                respond=False,
                stream=False
            )
            
            llm_response = ""
            for chunk in updated_entity:
                llm_response += chunk['response']
            # parse response
            updated_entity = GenEntity(
                name=name,
                description=llm_response.strip()
            )
            print("Added new entity:")
            print(llm_response)
            # add to updated list
            updated_list.entities.append(updated_entity)
        # now we have the final list
        self.entity_list = updated_list
        return updated_list
    
    def format_readable(self) -> str:
        """
        Convert entities to JSON for manual editing.
        """
        return self.entity_list.model_dump_json(indent=2)

    def import_readable(self, entity_data:str):
        """
        Import entity definitions from JSON.
        """
        self.entity_list = GenEntityList.model_validate_json(entity_data)

# a union type covering the possible entity manager types
# you can discriminate it by using Field(discriminator='entity_manager_class')
EntityManagerType = Union[EntityManager, SimpleEntityManager, JSONEntityManager]