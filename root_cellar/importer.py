import argparse
import asyncio
import sys
import uuid

from pathlib import Path
from root_cellar.entity import JSONEntityManager
from root_cellar.manager import StructuredHierarchicalManager, StructuredHierarchicalMemory, ChatThread

def parse_args() -> argparse.Namespace:
    """
    Set up command line argument parsing.
    """
    parser = argparse.ArgumentParser(description=(
        "Import a series of messages into root-cellar format. Automatically generates entities and summaries."
    ))
    parser.add_argument("-I", "--input", required=True, type=Path, action="store", help="Path to the input file to use")
    parser.add_argument("-O", "--output", required=True, type=Path, action="store", help="Path to the output file to store results.")
    parser.add_argument("-S", "--settings", required=True, type=Path, action="store", help="Path to the manager object which will be used to import settings.")
    parser.add_argument("-F", "--force", action="store_true", help="Overwrite the output file if it already exists.")
    return parser.parse_args()

async def main():
    # automatically parse the arguments
    args = parse_args()

    # try loading the topic file
    input_path = args.input
    
    if not input_path.exists():
        raise FileNotFoundError(f"Error: Input file not found: {input_path}")
    if not args.settings.exists():
        raise FileNotFoundError(f"Error: Settings file not found: {input_path}")
    
    # Read the input file
    try:
        chat_manager = StructuredHierarchicalManager.model_validate_json(json_data=input_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Read the settings file
    try:
        settings = StructuredHierarchicalManager.model_validate_json(json_data=args.settings.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Error reading settings file: {e}")
        sys.exit(1)
    
    # check if output file exists
    if args.output.exists() and not args.force:
        print(f"Output file already exists: {args.output}")
        sys.exit(1)
    
    # initialize session manager
    entity_manager = JSONEntityManager(
        llm=settings.chat_memory.summary_llm,
        prompt_entity_list=settings.chat_memory.entity_manager.prompt_entity_list,
        max_summary_depth=settings.chat_memory.entity_manager.max_summary_depth
    )
    chat_thread = ChatThread(session_id=str(uuid.uuid4()), system_prompt=chat_manager.chat_memory.chat_thread.system_prompt)
    chat_memory = StructuredHierarchicalMemory(
        summary_llm=settings.chat_memory.summary_llm,
        chat_thread=chat_thread,
        entity_manager=entity_manager,
        prop_ctx=settings.chat_memory.prop_ctx,
        prop_summary=settings.chat_memory.prop_summary,
        n_levels=settings.chat_memory.n_levels,
        n_tok_summarize=settings.chat_memory.n_tok_summarize
    )
    output_manager = StructuredHierarchicalManager(
        llm=settings.llm,
        chat_memory=chat_memory
    )
    print(output_manager.model_dump_json(indent=2))
    mem_len = 0
    for msg in chat_manager.chat_memory.chat_thread.archived_messages:
        clean_msg = {
            "role": msg['role'],
            "content": msg['content']
        }
        output_manager.append_message(clean_msg)
        await output_manager.chat_memory.update_all_memory()
        if len(output_manager.chat_memory.all_memory) > mem_len:
            print(output_manager.chat_memory.all_memory[-1]['content'])
            mem_len = len(output_manager.chat_memory.all_memory)
    for msg in chat_manager.chat_memory.chat_thread.messages:
        clean_msg = {
            "role": msg['role'],
            "content": msg['content']
        }
        output_manager.append_message(clean_msg)
        await output_manager.chat_memory.update_all_memory()
        if len(output_manager.chat_memory.all_memory) > mem_len:
            print(output_manager.chat_memory.all_memory[-1]['content'])
            mem_len = len(output_manager.chat_memory.all_memory)
    
    output_txt = output_manager.model_dump_json(indent=2)
    args.output.write_text(output_txt, encoding='utf-8')

if __name__ == "__main__":
    asyncio.run(main())