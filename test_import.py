print("Testing imports...")
try:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    print("Success! All imports worked correctly.")
except ImportError as e:
    print(f"Import error: {e}")
