from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.memory import ChatMemoryBuffer

def read_docs(args, Settings):
    documents = SimpleDirectoryReader(input_dir = args.filepath, recursive = True).load_data()
    service_context = ServiceContext.from_defaults(llm = Settings.llm, embed_model = Settings.embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context = service_context, embed_model = Settings.embed_model)
    memory = ChatMemoryBuffer.from_defaults(token_limit = args.memory_token_limit)

    return index, memory