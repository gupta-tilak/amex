import time
import psutil
import pandas as pd
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start timer
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Calculate metrics
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n=== Performance Report for {func.__name__} ===")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        print(f"Memory usage: {final_memory:.1f} MB (Δ {final_memory - initial_memory:+.1f} MB)")
        print(f"CPU usage: {psutil.cpu_percent()}%")
        
        return result
    return wrapper

def batch_process_dataframe(df, func, batch_size=50000):
    """Process large dataframes in batches"""
    results = []
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{total_batches}")
        result = func(batch)
        results.append(result)
    
    return pd.concat(results, ignore_index=True)
