import asyncio
from prisma import Prisma

async def query_products_from_r1_index(indexes):
    db = Prisma()
    await db.connect()

    product_asins = await db.r1dataindex.find_many(
        where={
            'id': {
                'in': [index + 1 for index in indexes]
            }
        }
    )

    product_asins_list = [product.dict()['asin'] for product in product_asins]

    product_details = await db.products.find_many(
        where={
            'asin': {
                'in': product_asins_list
            }
        }
    )

    res = [product.dict() for product in product_details]

    await db.disconnect()

    return res

if __name__ == '__main__':
    product_res = asyncio.run(query_products_from_r1_index([0,1,2]))

    assert product_res is not None
    print(product_res)
