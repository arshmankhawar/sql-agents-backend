from schema.retriever import get_retriever

# Use a concrete domain for the schema retriever
r = get_retriever("airport")
tests = [
    "revenue by region for 2025",
    "top customers by lifetime value",
    "low stock inventory items",
    "quarterly sales trends by product",
]
for t in tests:
    tables = r.retrieve(t)
    names = [x["table"] for x in tables]
    print(f"Task: {t}")
    print(f"  Tables: {names}")
    print()
