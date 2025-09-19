import click

try:
    from embeddings import generate_embedding, generate_completion
    from search import (
        search_scipy_cosine,
        search_faiss_l2,
        search_faiss_dot,
        search_canberra,
        compare_top_ids,
        fetch_chunk_by_id
    )
except Exception:
    from .embeddings import generate_embedding, generate_completion
    from .search import (
        search_scipy_cosine,
        search_faiss_l2,
        search_faiss_dot,
        search_canberra,
        compare_top_ids,
        fetch_chunk_by_id
    )


@click.group()
def cli():
    """Assistant CLI"""


# @click.command()  # Register subcommands cli.add_command(ask)
@cli.command()
@click.option(
    "--query", "-q", prompt="Your question",
    default="adult population in Ukraine", show_default=True,
    help="Your natural language question for the assistant."
)
@click.option(
    "--top_k", "-k", default=5, show_default=True,
    help="Number of top results to retrieve from the vector DB."
)
def ask(query: str, top_k: int):
    """
    Query the assistant with Retrieval-Augmented Generation (RAG) search.

    1. Convert the query to an embedding.
    2. Search the DB with multiple similarity metrics.
    3. Rank & merge results from all search methods.
    4. Fetch the best chunk(s) as context.
    5. Generate a detailed completion using the model.
    """
    # Step 1: Embed the query into a vector
    query_vector = generate_embedding(query)

    # Step 2: Run multiple search methods
    pg = search_scipy_cosine(query_vector, top_n=top_k)
    #display_results(pg, "Cosine Similarity (Postgres)")
    faiss_l2 = search_faiss_l2(query_vector, top_n=top_k)
    #display_results(faiss_l2, "Euclidean Distance (FAISS)")
    faiss_dot = search_faiss_dot(query_vector, top_n=top_k)
    #display_results(faiss_dot, "Dot Product (FAISS)")
    canberra = search_canberra(query_vector, top_n=top_k)
    #display_results(canberra, "Canberra Distance (Canberra)")

    # Step 3: Combine rankings (ensemble)
    top_ids = compare_top_ids(pg, faiss_l2, faiss_dot, canberra, top_n=top_k)
    if not top_ids:
        click.echo("No results found.")
        return

    # Step 4: Fetch top context chunks from DB
    # context_chunks = []
    # for cid, group_id, score in top_ids:
    #     text = fetch_chunk_by_id(cid)
    #     if text:
    #         context_chunks.append(text)

    # context_text = "\n\n".join(context_chunks)
    best_id = top_ids[0][0]
    source_link = top_ids[0][-1]
    context = fetch_chunk_by_id(best_id)
    if context:
        click.echo(f"\n[Best Match: ID {best_id} link {source_link}]")
    else:
        click.echo("Best chunk not found in DB.")

    # Step 5: Generate model response
    # click.echo(f"\nAssistant:\n{answer}")
    click.echo("\n=== Assistant Answer ===\n")
    # Call RAG model
    answer = generate_completion(query, context)
    click.echo(answer)


if __name__ == "__main__":
    cli()