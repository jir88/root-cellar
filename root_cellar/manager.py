import re
from typing import List, Optional, Dict, Any, AnyStr, ClassVar
from pydantic import BaseModel, Field, SerializeAsAny
from abc import ABC


from root_cellar.llm import OpenAILLM,LLMType
from root_cellar.entity import SimpleEntityManager,JSONEntityManager

class ChatThread(BaseModel):
    """
    A single chat thread between a user and an LLM.
    
    Attributes:
    session_id (str): The unique identifier of the chat thread.
    messages (list): A list of messages sent in the chat thread.
    """

    # pulls role names out of a string representation of a thread
    role_regex: ClassVar[re.Pattern[AnyStr]] = re.compile(r"{{(.+?)}}")

    session_id: str = Field(..., description="The unique identifier of this chat thread.")
    system_prompt: Optional[str] = Field(None, description="An optional system prompt for this chat thread.")
    messages: List[Dict[str, Any]] = Field(
        default=[],
        description="A list of dicts representing the messages in this thread."
    )
    archived_messages: List[Dict[str, Any]] = Field(
        default=[],
        description="A list of dicts representing archived past messages which are no longer in the context window."
    )
    user_role: str = Field(
        default="user", 
        description="The name of the human/user role in this chat. Defaults to 'user'."
    )
    ai_role: str = Field(
        default="assistant", 
        description="The name of the AI/assistant role in this chat. Defaults to 'assistant'."
    )

    def archive_messages(self, start_idx, stop_idx):
        """
        Move some messages from the main message thread to the
        archived messages list.

        Args:
        start_idx (int): index of first message to archive (inclusive)
        end_idx (int): index of last message to archive (exclusive)
        """
        self.archived_messages.extend(self.messages[start_idx:stop_idx])
        del self.messages[start_idx:stop_idx]

    def format_readable(self):
        """
        Convert all messages in this thread into a human-readable and editable
        format. Message roles are displayed in curly brackets: {{role}} with
        message text following. Leading and trailing whitespace are ignored.
        """
        result = ""
        for i in range(0, len(self.messages)):
            result += "{{" + self.messages[i]['role'] + "}}\n" + self.messages[i]['content'] + "\n"
        return result

    def import_readable(self, formatted_messages:str):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        formatted_messages (str): The formatted messages to be parsed.
        """
        # splitting with capturing groups returns the roles too
        msg_parts = self.role_regex.split(formatted_messages)
        # drop first item, which is blank for some reason
        msg_parts = msg_parts[1:]
        
        # strip out extra whitespace and format
        # should probably pre-allocate this...
        parsed_messages = []
        for i in range(0, len(msg_parts), 2):
            msg_dict = {
                "id": len(self.archived_messages) + i/2,
                "role": msg_parts[i],
                "content": str.strip(msg_parts[i + 1])
            }
            parsed_messages.append(msg_dict)
        self.messages = parsed_messages

    # def to_json(self):
    #     """
    #     Write this object out as a JSON object.

    #     Returns: a string containing the JSON object
    #     """
    #     # define state to save
    #     settings_to_download = {"session_id": self.session_id,
    #                             "system_prompt": self.system_prompt,
    #                             "messages": self.messages,
    #                             "user_role": self.user_role,
    #                             "ai_role": self.ai_role,
    #                             "archived_messages": self.archived_messages
    #                             }
    #     # dump it to a JSON file
    #     return json.dumps(settings_to_download)

    # @classmethod
    # def from_json(cls, json_data):
    #     """
    #     Load saved session state from a JSON object.
    #     Args:
    #     json_data (str): JSON object or file containing session data

    #     Returns: a new ChatSession object initialized from the JSON data
    #     """
    #     # load saved state
    #     if type(json_data) == str:
    #         uploaded_settings = json.loads(json_data)
    #     else:
    #         uploaded_settings = json.load(json_data)
    #     # create new thread object
    #     new_obj = cls(session_id=uploaded_settings.get('session_id'))
    #     # load system prompt
    #     new_obj.system_prompt = uploaded_settings.get('system_prompt')
    #     # load messages
    #     new_obj.messages = uploaded_settings.get('messages')
    #     # load user role
    #     new_obj.user_role = uploaded_settings["user_role"]
    #     # load AI role
    #     new_obj.ai_role = uploaded_settings["ai_role"]
    #     # load archived messages
    #     new_obj.archived_messages = uploaded_settings.get('archived_messages')
    #     # return object
    #     return new_obj

class ChatMemory(ABC, BaseModel):
    """
    Abstract class for various methods of helping LLMs 'remember' information beyond
    their context lengths.
    """

    chat_thread: ChatThread = Field(
        default=...,
        description="The chat thread associated with this memory object."
    )

    class Config:
        arbitrary_types_allowed = True

    def add_documents(self, docs):
        """
        Add a set of documents to memory. How these are added will vary by implementation.

        Args:
        docs (list[dict]): a list of dicts, where each dict has the document ('content' key)
            plus any other useful metadata.
        """
        raise NotImplementedError("Method not implemented!")

    def update_memory(self, mem_id, mem_content):
        raise NotImplementedError("Method not implemented!")

    def query_memory(self, query_text, n_results=3):
        raise NotImplementedError("Method not implemented!")

    def update_all_memory(self):
        raise NotImplementedError("Method not implemented!")

class HierarchicalSummaryMemory(ChatMemory):
    """
    Manages chat memory using a similar mechanism to the one used by perchance.ai.
    The memory structure uses multiple 'levels' of summaries, with the raw messages
    treated as level 0. Each time the length of a summary level plus all higher levels
    exceeds some set proportion of the context window, part of that summary level is
    summarized and added to the next higher summary level. This process is applied to
    all summary levels as needed. The top level is a single summary message. Lower 
    level summaries are merged into this one instead of adding more.

    This class keeps track of all message summaries, the messages that they summarized,
    and the chat message index where the last summarized message is located.
    """

    summary_llm: SerializeAsAny[LLMType] = Field(
        default=...,
        discriminator='llm_class',
        description="The LLM model to use when generating summaries. NOTE: make sure this model has the same allocated context window size as the main LLM!"
    )
    entity_manager: SerializeAsAny[SimpleEntityManager] = Field(
        default=...,
        description="The entity manager object associated with this memory object."
    )
    summary_prompt: str = Field(
        default="""You are summarizing a long series of messages into a concise but accurate summary. You will be given any relevant prior context and the user will provide the messages to be summarized. You must only summarize the content of the messages themselves, not the prior context. Make sure to include all important details.

        Prior context:
        {context}

        Now the user will provide you with the messages to be summarized. Respond only with a single-paragraph summary, no additional commentary.""",
        description="Optional custom summarization prompt. To include prior \
            context in the prompt, use placeholder {context}. If no custom prompt \
            provided, uses a default prompt instead."
    )
    prop_ctx: float = Field(
        default=0.8,
        description="The proportion of the total context window that summaries \
            plus un-summarized messages may use up before triggering a higher-level \
            summary."
    )
    prop_summary: float = Field(
        default=0.5,
        description="The proportion of a message/summary level that can \
            be occupied by messages/summaries of higher level. Each summary \
            level is allocated prop_summary of the context alloted to the next higher \
            level (total context window for original thread messages)."
    )
    n_levels: int = Field(
        default=3,
        description="The maximum number of summary levels to use."
    )
    n_tok_summarize: int = Field(
        default=1024,
        description="The target number of tokens to summarize in one pass. \
            If this corresponds to less than one message, that whole message will be \
            summarized."
    )
    all_memory: List[Dict[str, Any]] = Field(
        default=[],
        description="Summaries stored as a list of dicts containing summary level, the actual \
            messages (or lower-level summaries) that were summarized, and the index \
            of the final summarized message in the full chat thread"
    )
    archived_memory: List[Dict[str, Any]] = Field(
        default=[],
        description="Summaries which have been collapsed into the top-level summary."
    )

    def update_all_memory(self):
        """
        Update memory so that all message levels fit within their corresponding
        token allotments. If the raw messages themselves are too big, the oldest
        messages will be summarized and archived. Note that this process only 
        summarizes the oldest n_tok_summarize tokens of each level (rounded up to
        the next message), so one or more levels may still be 'over-budget' afterwards.

        This function also updates the entity list or creates a new one if none is present.
        """
        # first memory will be in the highest current level
        if len(self.all_memory) > 0:
            current_level = self.all_memory[0]['level']
        else:
            # no memories, so we're at level 0 (raw messages)
            current_level = 0
        
        # tokens occupied by higher levels of summary
        higher_level_tokens = 0
        
        # the index of the first summary we will be summarizing
        # starts at 0, then updated as we finish handling each level
        start_summ_index = 0

        # now iterate through the levels until we hit the raw message level
        while current_level > 0:
            # find index of first summary in this level
            start_summ_index = self._get_index_of_first_summary_in_level(level=current_level)
            # is this level too big?
            level_allowance = self.summary_llm.sampling_options['num_ctx']*self.prop_ctx*self.prop_summary**current_level
            current_level_tokens = self.summary_level_size(level=current_level)
            if (higher_level_tokens + current_level_tokens) >= level_allowance:
                # if the next-to-highest summary level is too big, we include the top level summary
                if current_level == (self.n_levels - 1) and self.all_memory[0]['level'] == self.n_levels:
                    # summarizing both the existing top-level summary and part of the
                    # next level down into a new top-level summary
                    idx_to_summarize = [0]
                    idx_to_summarize.extend(self._get_summary_indices_in_level(level=current_level))
                else:
                    # normally just summarizing within the current level
                    idx_to_summarize = self._get_summary_indices_in_level(level=current_level)
                # get messages within level that we are going to summarize
                lim_idx = self._get_messages_with_token_size(
                    msgs=[self.all_memory[i] for i in idx_to_summarize],
                    n_tok=self.n_tok_summarize
                )
                idx_to_summarize = idx_to_summarize[slice(lim_idx+1)]
                # summarize the messages
                summarized_messages = [self.all_memory[i] for i in idx_to_summarize]
                new_top_summary = self._summarize_messages(
                    messages=summarized_messages,
                    prior_summaries=self.all_memory[:start_summ_index]
                )
                # don't update entity list here -- we've already ingested that data
                # when summarizing the raw messages
                # put old summary messages in archive
                self.archived_memory.extend(summarized_messages)
                # delete from active summaries
                # have to work in reverse order so indices don't change while deleting
                for i in reversed(idx_to_summarize):
                    del self.all_memory[i]
                # insert new summary
                nts_dict = {
                    # make sure updated top-level summaries keep the same level
                    'level': min(current_level + 1, self.n_levels),
                    # last message index of the last summary in this summary
                    'msg_idx': max([s['msg_idx'] for s in summarized_messages]),
                    'content': new_top_summary
                }
                self.all_memory.insert(
                    # replace the first summarized index
                    idx_to_summarize[0],
                    nts_dict
                )
            # add current level's remaining tokens to the cumulative total
            higher_level_tokens += self.summary_level_size(level=current_level)
            # move to next level down
            current_level -= 1

        # now we look at the raw messages
        # index of first summary in this level is the end of the memory list
        start_summ_index = len(self.all_memory)
        # is message level too big?
        level_allowance = self.summary_llm.sampling_options['num_ctx']*self.prop_ctx
        # how long is the current message thread?
        current_level_tokens = 0
        for summary in self.chat_thread.messages:
            current_level_tokens += self._chars_to_tokens(summary['content'])
        # if it is too big
        if (higher_level_tokens + current_level_tokens) >= level_allowance:
            # index of last message that fills up our summarization budget
            lim_idx = self._get_messages_with_token_size(
                msgs=self.chat_thread.messages,
                n_tok=self.n_tok_summarize
            )
            summarized_messages = self.chat_thread.messages[slice(lim_idx+1)]
            # summarize
            new_top_summary = self._summarize_messages(
                messages=summarized_messages,
                prior_summaries=self.all_memory
            )
            # update entity list
            self._update_entity_definitions(
                messages=summarized_messages,
                prior_summaries=self.all_memory[:start_summ_index]
            )
            # archive these messages from the chat thread
            self.chat_thread.archive_messages(
                start_idx=0,
                stop_idx=lim_idx+1
            )
            # insert new first-level summary
            nts_dict = {
                'level': 1,
                # last message index of the last message in this summary
                'msg_idx': round(summarized_messages[-1]['id']),
                'content': new_top_summary
            }
            # new summary goes at the end
            self.all_memory.append(nts_dict)

    def _summarize_messages(self, messages:list, prior_summaries:list=[]):
        """
        Summarize a list of messages, optionally including a list of older summaries
        as context.

        Args:
        messages (list): a list of messages to summarize
        prior_summaries (list): a list of older summaries to be used as context when summarizing

        Returns: the summary
        """
        # if no prior context, just put 'None' in as a placeholder
        if len(prior_summaries) == 0:
            prior_summaries = [{ 'content': "No prior context." }]
        
        # construct system prompt
        sys_prompt = {
            'role': 'system',
            'content': self.summary_prompt.format(context="\n\n".join([ps['content'] for ps in prior_summaries]))
        }
        user_prompt = {
            'role': 'user',
            'content': "Please summarize the following messages:\n\n" + "\n\n".join([m['content'] for m in messages])
        }
        # generate the summary
        llm_response = self.summary_llm.generate_instruct(
            messages=[sys_prompt, user_prompt],
            respond=True,
            response_role="assistant",
            stream=False
        )
        # pull the first/only result off the generator and strip whitespace
        return next(llm_response)['response'].strip()
    
    def _update_entity_definitions(self, messages: List[Dict[str, str]], prior_summaries: List[Dict[str, str]]):
        """
        Update any relevant entities. Extend this method to provide more complex handling.
        """
        self.entity_manager.update_entities(
                messages=messages,
                prior_summaries=prior_summaries
            )

    def _get_index_of_first_summary_in_level(self, level:int):
        if len(self.all_memory) == 0:
            return 0
        for i in range(len(self.all_memory)):
            if self.all_memory[i]['level'] == level:
                return i
        # no summaries at this level
        return -1

    def _get_summaries_in_level(self, level:int):
        """
        Get all the summaries of a given level, in order.

        Returns: a list of all the summaries in that level
        """
        if len(self.all_memory) == 0:
            return []
        level_msgs = []
        for summary in self.all_memory:
            if summary['level'] == level:
                level_msgs.append(summary)
        return level_msgs
    
    def _get_summary_indices_in_level(self, level:int):
        """
        Get the indices of all the summaries of a given level, in order.

        Returns: a list of the summary indicies in the memory list
        """
        if len(self.all_memory) == 0:
            return []
        level_idx = []
        for idx in range(len(self.all_memory)):
            if self.all_memory[idx]['level'] == level:
                level_idx.append(idx)
        return level_idx

    def _get_messages_with_token_size(self, msgs, n_tok:int):
        """
        Calculates the index of the first message in this list in which
        the cumulative number of tokens exceeds n_tok, or len(msgs) if
        the total size of all messages is less than n_tok.
        """
        cum_tokens = 0

        for i in range(len(msgs)):
            cum_tokens += self._chars_to_tokens(text=msgs[i]['content'])
            if cum_tokens >= n_tok:
                # this is the message where we go over the token limit
                return i
        return len(msgs)
    
    def summary_level_size(self, level:int):
        """
        Estimate the number of tokens in all summaries of a given level.

        Args:
        level (int): the summary level of interest

        Returns: the approximate number of tokens.
        """
        if len(self.all_memory) == 0:
            return 0
        level_size = 0
        for summary in self.all_memory:
            if summary['level'] == level:
                level_size += self._chars_to_tokens(summary['content'])
        return level_size
    
    def _chars_to_tokens(self, text:str):
        """
        Extremely rough estimation of tokens in a string (~3.5 chars/token).

        Args:
        text (str): the text to be estimated

        Returns: the approximate number of tokens
        """
        if isinstance(self.summary_llm, OpenAILLM):
            return self.summary_llm.count_tokens(text)
        return max(1, len(text)/3.5)

    def format_readable(self):
        """
        Convert all active summaries into a human-readable and editable
        format. Summary levels and positions are displayed in curly brackets:
        {{L<level>@<position>}} with summary text following. Leading and trailing
        whitespace are stripped.
        """
        result = ""
        for mem in self.all_memory:
            result += "{{L" + str(mem['level']) + "@" + str(mem['msg_idx']) + "}}\n" + mem['content'] + "\n"
        return result
    
    memory_regex: ClassVar[re.Pattern[AnyStr]] = re.compile(r"{{L(\d+)@(\d+)}}")

    def import_readable(self, formatted_messages:str):
        """
        Parse messages exported by format_readable and use them to replace any
        existing messages in this chat session.

        Args:
        formatted_messages (str): The formatted messages to be parsed.
        """
        # splitting with capturing groups returns the roles too
        msg_parts = self.memory_regex.split(formatted_messages)
        # drop first item, which is blank for some reason
        msg_parts = msg_parts[1:]
        
        # strip out extra whitespace and format
        # should probably pre-allocate this...
        parsed_messages = []
        for i in range(0, len(msg_parts), 3):
            msg_dict = {
                "level": int(msg_parts[i]),
                "msg_idx": int(msg_parts[i + 1]),
                "content": str.strip(msg_parts[i + 2])
            }
            parsed_messages.append(msg_dict)
        self.all_memory = parsed_messages

class StructuredHierarchicalMemory(HierarchicalSummaryMemory):
    """
    Manages chat memory using the same hierarchical method as the superclass.
    Entity management uses pydantic to generate and load entities.
    """
    
    entity_manager: SerializeAsAny[JSONEntityManager] = Field(
        default=...,
        description="The entity manager object associated with this memory object."
    )

class StatefulChatManager(ABC, BaseModel):
    """
    Top-level class managing all the moving parts of a stateful chat.
    """

    llm: SerializeAsAny[LLMType] = Field(
        default=...,
        discriminator='llm_class',
        description="The LLM instance to use for generating responses."
    )
    chat_memory: SerializeAsAny[ChatMemory] = Field(
        default=...,
        description="Memory instance to track long-term memory of the chat thread. \
            NOTE: ensure that this memory is managing the same object passed as chat_thread!"
    )

    def append_message(self, message):
        """
        Append message.

        Args:
        message (dict): Dict containing at least 'role' and 'content' keys
        """
        ct = self.chat_memory.chat_thread
        # if missing, set ID to be the message index
        if "id" not in message:
            message['id'] = len(ct.messages) + len(ct.archived_messages)
        # add to the active chat thread
        ct.messages.append(message)
        # Needs updating: embed regenerated AI response
        #st.session_state.chat_session.embed_text(st.session_state.chat_session.messages[-1], "message")

    def messages_to_memory(self, n_msgs):
        """
        Remove a number of the oldest messages from context and commit them
        to memory.

        Args:
        n_msgs (int): Number of oldest messages
        """
        ct = self.chat_memory.chat_thread
        # pull messages
        old_msgs = ct.messages[0:n_msgs]
        # add to memory. Send system prompt or the model goes insane
        self.chat_memory.add_messages(old_msgs, context=self.compile_system_prompt())
        # archive messages in thread
        ct.archive_messages(0, n_msgs)

    def export_thread(self):
        raise NotImplementedError("Not implemented!")

    def import_thread(self, messages):
        """
        Import a thread formatted as text. Existing messages are altered to
        reflect differences with the imported text.
        """
        raise NotImplementedError("Not implemented!")

    def compile_system_prompt(self):
        """
        Combine raw prompt, the most recent message summary, and the entity list
        into a full system prompt.

        TODO: the memory stuff should probably be delegated to the memory class.
        """
        ct = self.chat_memory.chat_thread
        # start with the system prompt for the current chat thread, if any
        full_sys_prompt = ""
        if ct.system_prompt is not None:
            full_sys_prompt += ct.system_prompt.strip()
        # add top-level summary from memory
        if self.chat_memory.full_summary is not None:
            full_sys_prompt += "\n\nComplete summary of all previous messages:\n" + self.chat_memory.full_summary
        # add entity list, if any
        if self.chat_memory.entity_list is not None:
            full_sys_prompt += "\n\nEntitites mentioned previously:\n" + self.chat_memory.entity_list
        # add latest message summary, if any
        if len(self.chat_memory.message_summaries) > 0:
            full_sys_prompt += "\n\nSummary of recent previous messages:\n" + self.chat_memory.message_summaries[-1]['content']
        return full_sys_prompt
    
    def get_response(self, stream=True):
        """
        Generate an AI response starting with the end of the current thread.
        Uses summary messages to ensure we don't overflow the context window.

        Args:
        stream (bool): whether to stream the response or not

        Returns: a generator if streaming or the response text if not streaming
        """
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_memory.chat_thread.messages)
        # generate response using current thread's AI role
        return self.llm.generate_instruct(messages=all_msgs,
                                          stream=stream
                                          )

    def continue_response(self, stream=True):
        """
        Continue generating from the end of the most recent message.
        """
        # make the system prompt
        sys_prompt = self.compile_system_prompt().strip()
        all_msgs = [{ 'role': "system", 'content': sys_prompt }]
        # add in-context messages after sys prompt
        all_msgs.extend(self.chat_memory.chat_thread.messages)
        # continue generating from end of last message
        return self.llm.generate_instruct(messages=all_msgs,
                                          respond=False,
                                          stream=stream
                                          )

class HierarchicalSummaryManager(StatefulChatManager):
    """
    Custom chat manager that compresses long chats into the context window using 
    heirarchical summaries of older messages, similar to the method used by
    perchance.ai.
    """

    chat_memory: SerializeAsAny[HierarchicalSummaryMemory] = Field(
        default=...,
        description="Hierarchical memory instance to track long-term memory of the chat thread."
    )

    def compile_system_prompt(self):
        """
        Combine raw prompt and summaries into a full system prompt.
        """
        ct = self.chat_memory.chat_thread
        # start with the system prompt for the current chat thread, if any
        full_sys_prompt = ""
        if ct.system_prompt is not None:
            full_sys_prompt += ct.system_prompt.strip()
        # add entity list, if any
        entity_list = self.chat_memory.entity_manager.entity_list
        if entity_list is not None and len(entity_list) > 0:
            full_sys_prompt += "\n\nEntities appearing in previous messages:\n" + entity_list
        # add top-level summary from memory
        if len(self.chat_memory.all_memory) > 0:
            mems = [m['content'] for m in self.chat_memory.all_memory]
            full_sys_prompt += "\n\nSummary of all previous messages:\n" + "\n".join(mems)
        return full_sys_prompt

class StructuredHierarchicalManager(HierarchicalSummaryManager):
    """
    Custom chat manager that compresses long chats into the context window using 
    heirarchical summaries of older messages, similar to the method used by
    perchance.ai.

    This manager uses JSON-formatted generation to provide more fine-grained control
    over entity creation and management.
    """

    chat_memory: SerializeAsAny[StructuredHierarchicalMemory] = Field(
        default=...,
        description="Hierarchical memory instance to track long-term memory of the chat thread."
    )

    def compile_system_prompt(self):
        """
        Combine raw prompt and summaries into a full system prompt.
        """
        ct = self.chat_memory.chat_thread
        # start with the system prompt for the current chat thread, if any
        full_sys_prompt = ""
        if ct.system_prompt is not None:
            full_sys_prompt += ct.system_prompt.strip()
        # add entity list, if any
        entity_list = self.chat_memory.entity_manager.entity_list.entities
        if entity_list is not None and len(entity_list) > 0:
            ent_txt = "\n\n".join([ent.name + ": " + ent.description for ent in entity_list])
            full_sys_prompt += "\n\nEntities appearing in previous messages:\n" + ent_txt
        # add top-level summary from memory
        if len(self.chat_memory.all_memory) > 0:
            mems = [m['content'] for m in self.chat_memory.all_memory]
            full_sys_prompt += "\n\nSummary of all previous messages:\n" + "\n".join(mems)
        return full_sys_prompt