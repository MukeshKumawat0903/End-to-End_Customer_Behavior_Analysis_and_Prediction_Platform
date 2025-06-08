import duckdb
import pandas as pd

def load_ecommerce_data(file_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load eCommerce data from parquet file with DuckDB"""
    con = duckdb.connect()
    
    # Get columns and filter nulls
    columns = con.execute(f"DESCRIBE SELECT * FROM '{file_path}'").fetchdf()['column_name'].tolist()
    null_filter = " AND ".join([f"{col} IS NOT NULL" for col in columns])
    
    # Get top categories filter
    top_categories = con.execute(f"""
        SELECT category_code
        FROM '{file_path}'
        WHERE category_code IS NOT NULL
        GROUP BY category_code
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """).fetchdf()['category_code'].tolist()
    
    # Execute final query
    limit_clause = f"LIMIT {sample_size}" if sample_size is not None else ""
    query = f"""
        SELECT *
        FROM '{file_path}'
        WHERE {null_filter}
          AND category_code IN ({', '.join([f"'{cat}'" for cat in top_categories])})
        {limit_clause}
    """
    
    return con.execute(query).fetchdf()