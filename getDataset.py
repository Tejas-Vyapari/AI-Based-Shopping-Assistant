import pandas as pd
import numpy as np
from random import randint
from collections import Counter
from typing import List, Dict


def get_data(history_size: int = 1000):
    search_history = [
        'shirt', 'watch', 'wallet', 'shoes', 'belt', 'rice', 'sugar', 'skirt',
        'sandal', 'heals', 'handbag', 'bagpack', 'pillow', 'teddy', 'suitcase',
        'mobiles', 'laptop', 'fridge', 'desk', 'soda', 'hat', 'bangles', 
        'earrings', 'necklace', 'ring', 'shampoo', 'dates', 'cashew', 'juice', 
        'lays', 'maggi', 'keyboard', 'paper', 'pen', 'pencil', 'radio', 
        'rocket fuel', 'bulbs', 'ink', 'pajamas', 'shower', 'letters', 'books', 
        'bottles', 'cars', 'bread', 'chain', 'bullets', 'mutton', 'cakes', 
        'cards', 'pearls', 'rope', 'bacon', 'tomatoes', 'eggs', 'onions', 
        'yeast', 'vegetables', 'baking powder', 'cheese', 'milk', 'handbags', 
        'boots', 'sweaters', 'dress', 'jacket'
    ]

    item_count = len(search_history)
    item_ids = [randint(0, item_count - 1) for _ in range(history_size)]
    item_names = [search_history[item_id] for item_id in item_ids]
    timestamps = [randint(1550350815, 1550370815) for _ in range(history_size)]

    # Create DataFrame
    df = pd.DataFrame({
        'Item': item_ids,
        'Item_names': item_names,
        'Time': timestamps
    })

    df.to_csv('dataset.csv', index=False)
    print("Generated Dataset:\n", df.head())


def calculate_frequency():
    # Check if file exists
    try:
        df = pd.read_csv('dataset.csv')
    except FileNotFoundError:
        print("Dataset not found. Please generate it using `get_data()`.")
        return

    frequency = Counter(df['Item'])
    item_list = df[['Item', 'Item_names']].drop_duplicates().set_index('Item')
    
    # Calculate frequency as a DataFrame
    frequency_df = pd.DataFrame.from_dict(frequency, orient='index', columns=['Frequency'])
    frequency_df = frequency_df.merge(item_list, left_index=True, right_index=True)
    frequency_df = frequency_df.reset_index().rename(columns={'index': 'Item'})

    recency = calculate_recency(df)
    frequency_df['Recency'] = recency

    frequency_df.to_csv('person.csv', index=False)
    print("Frequency and Recency Calculation Saved:\n", frequency_df.head())


def calculate_recency(df: pd.DataFrame) -> List[float]:
    # Calculate recency per item
    latest_time = df['Time'].max()
    recency = (
        df.groupby('Item')['Time']
        .apply(lambda times: np.mean([latest_time - t for t in times]))
        .tolist()
    )
    return recency


if __name__ == "__main__":
    get_data()
    calculate_frequency()

