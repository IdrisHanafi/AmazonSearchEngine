import asyncio
from prisma import Prisma
import pandas as pd

async def main() -> None:
    db = Prisma()
    await db.connect()

    chunks = pd.read_json(
        "datasets/meta_Electronics.json", 
        lines=True, 
        chunksize=1000
    )

    for chunk_df in chunks:
        chunk_df = chunk_df[['asin', 'title', 'description', 'price', 'brand', 'imageURL', 'category']].to_dict('records')
        product = await db.products.create_many(
            data=chunk_df,
            skip_duplicates=True
        )
        # break

    await db.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
