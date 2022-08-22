import asyncio
from prisma import Prisma
import pandas as pd

async def insert_main_product_info() -> None:
    db = Prisma()
    await db.connect()

    chunksize = 1000
    chunks = pd.read_json(
        "datasets/meta_Electronics.json", 
        lines=True, 
        chunksize=chunksize
    )

    i = 1
    for chunk_df in chunks:
        chunk_df = chunk_df[['asin', 'title', 'description', 'price', 'brand', 'imageURL', 'category']].to_dict('records')
        product = await db.products.create_many(
            data=chunk_df,
            skip_duplicates=True
        )
        print(f"{i*chunksize} inserted")
        i += 1
        # break

    await db.disconnect()

async def insert_r1_indexed() -> None:
    db = Prisma()
    await db.connect()

    chunksize = 1000
    chunks = pd.read_json(
        "datasets/r1_data/R1_data_indexed.json",
        lines=True,
        chunksize=chunksize
    )

    i = 1
    for chunk_df in chunks:
        chunk_df = chunk_df[['asin', 'title', 'category_list']].to_dict('records')
        product = await db.r1dataindex.create_many(
            data=chunk_df
        )
        print(f"{i*chunksize} inserted")
        i += 1

    await db.disconnect()

if __name__ == '__main__':
    asyncio.run(insert_main_product_info())
    asyncio.run(insert_r1_indexed())
